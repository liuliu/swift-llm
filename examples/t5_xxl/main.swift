import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let torch = Python.import("torch")
let transformers = Python.import("transformers")

print(transformers.T5Tokenizer)

let tokenizer = transformers.T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
let model = transformers.T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-xxl", device_map: "auto")
print(model)

let input_text = "Expand the following prompt to add more detail: A man."
let input_ids = tokenizer(input_text, return_tensors: "pt").input_ids.to("cuda")
print(input_ids)
let state_dict = model.state_dict()

print(state_dict.keys())

let outputs = model.generate(input_ids, max_new_tokens: 1)
print(tokenizer.decode(outputs[0]))

func T5TextEmbedding(vocabularySize: Int, embeddingSize: Int) -> Model {
  let tokenEmbed = Embedding(
    Float.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  return tokenEmbed
}

func T5LayerSelfAttention(k: Int, h: Int, b: Int, t: Int, outFeatures: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let causalAttentionMask = Input()
  let positionBias = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  // No scaling the queries.
  let queries = toqueries(x).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
  dot = dot + positionBias
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: outFeatures, noBias: true)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, causalAttentionMask, positionBias], [out]))
}

func T5DenseGatedActDense(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model) {
  let x = Input()
  let wi_0 = Dense(count: intermediateSize, noBias: true)
  let wi_1 = Dense(count: intermediateSize, noBias: true)
  var out = wi_1(x) .* wi_0(x).GELU(approximate: .tanh)
  let wo = Dense(count: hiddenSize, noBias: true)
  out = wo(out)
  return (wi_0, wi_1, wo, Model([x], [out], name: "ff"))
}

func T5Block(prefix: String, k: Int, h: Int, b: Int, t: Int, outFeatures: Int, intermediateSize: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let causalAttentionMask = Input()
  let positionBias = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1])
  let (tokeys, toqueries, tovalues, unifyheads, attention) = T5LayerSelfAttention(k: k, h: h, b: b, t: t, outFeatures: outFeatures)
  var out = x + attention(norm1(x), causalAttentionMask, positionBias)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1])
  let (wi_0, wi_1, wo, ff) = T5DenseGatedActDense(hiddenSize: outFeatures, intermediateSize: intermediateSize)
  out = out + ff(norm2(out))
  let reader: (PythonObject) -> Void = { state_dict in
    let layer_0_layer_norm_weight = state_dict["\(prefix).layer.0.layer_norm.weight"].cpu().float().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: layer_0_layer_norm_weight))
    let k_weight = state_dict["\(prefix).layer.0.SelfAttention.k.weight"].cpu().float().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    let q_weight = state_dict["\(prefix).layer.0.SelfAttention.q.weight"].cpu().float().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    let v_weight = state_dict["\(prefix).layer.0.SelfAttention.v.weight"].cpu().float().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    let o_weight = state_dict["\(prefix).layer.0.SelfAttention.o.weight"].cpu().float().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: o_weight))
    let layer_1_layer_norm_weight = state_dict["\(prefix).layer.1.layer_norm.weight"].cpu().float().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: layer_1_layer_norm_weight))
    let wi_0_weight = state_dict["\(prefix).layer.1.DenseReluDense.wi_0.weight"].cpu().float().numpy()
    wi_0.weight.copy(from: try! Tensor<Float>(numpy: wi_0_weight))
    let wi_1_weight = state_dict["\(prefix).layer.1.DenseReluDense.wi_1.weight"].cpu().float().numpy()
    wi_1.weight.copy(from: try! Tensor<Float>(numpy: wi_1_weight))
    let wo_weight = state_dict["\(prefix).layer.1.DenseReluDense.wo.weight"].cpu().float().numpy()
    wo.weight.copy(from: try! Tensor<Float>(numpy: wo_weight))
  }
  return (reader, Model([x, causalAttentionMask, positionBias], [out]))
}

func T5ForConditionalGeneration() -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let causalAttentionMask = Input()
  let relativePositionBucket = Input()
  let textEmbed = T5TextEmbedding(vocabularySize: 32_128, embeddingSize: 4_096)
  var out = textEmbed(x)
  let relativePositionEmbedding = Embedding(Float.self, vocabularySize: 32, embeddingSize: 64)
  let positionBias = relativePositionEmbedding(relativePositionBucket).reshaped([1, 13, 13, 64]).permuted(0, 3, 1, 2)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = T5Block(prefix: "encoder.block.\(i)", k: 64, h: 64, b: 1, t: 13, outFeatures: 4_096, intermediateSize: 10_240)
    out = block(out, causalAttentionMask, positionBias)
    readers.append(reader)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1])
  out = finalNorm(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["shared.weight"].cpu().float().numpy()
    textEmbed.weight.copy(from: try! Tensor<Float>(numpy: vocab))
    let relative_attention_bias_weight = state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].cpu().float().numpy()
    relativePositionEmbedding.weight.copy(from: try! Tensor<Float>(numpy: relative_attention_bias_weight))
    for reader in readers {
      reader(state_dict)
    }
    let final_layer_norm_weight = state_dict["encoder.final_layer_norm.weight"].cpu().float().numpy()
    finalNorm.weight.copy(from: try! Tensor<Float>(numpy: final_layer_norm_weight))
  }
  return (reader, Model([x, causalAttentionMask, relativePositionBucket], [out]))
}

let graph = DynamicGraph()
let tokensTensor = graph.variable(.CPU, .C(13), of: Int32.self)
tokensTensor[0] = 28415
tokensTensor[1] = 8
tokensTensor[2] = 826
tokensTensor[3] = 9005
tokensTensor[4] = 12
tokensTensor[5] = 617
tokensTensor[6] = 72
tokensTensor[7] = 2736
tokensTensor[8] = 10
tokensTensor[9] = 71
tokensTensor[10] = 388
tokensTensor[11] = 5
tokensTensor[12] = 1

graph.withNoGrad {
  let (reader, textModel) = T5ForConditionalGeneration()
  let causalAttentionMask = graph.variable(.CPU, .NCHW(1, 1, 13, 13), of: Float.self)
  causalAttentionMask.full(0)
  /*
  for i in 0..<12 {
    for j in (i + 1)..<13 {
      causalAttentionMask[0, 0, i, j] = -Float.greatestFiniteMagnitude
    }
  }
  */
  let relativePositionBucket = Tensor<Int32>([
    0, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24, 24, 25,
    1,  0, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24, 24,
    2,  1,  0, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24,
    3,  2,  1,  0, 17, 18, 19, 20, 21, 22, 23, 24, 24,
    4,  3,  2,  1,  0, 17, 18, 19, 20, 21, 22, 23, 24,
    5,  4,  3,  2,  1,  0, 17, 18, 19, 20, 21, 22, 23,
    6,  5,  4,  3,  2,  1,  0, 17, 18, 19, 20, 21, 22,
    7,  6,  5,  4,  3,  2,  1,  0, 17, 18, 19, 20, 21,
    8,  7,  6,  5,  4,  3,  2,  1,  0, 17, 18, 19, 20,
    8,  8,  7,  6,  5,  4,  3,  2,  1,  0, 17, 18, 19,
    8,  8,  8,  7,  6,  5,  4,  3,  2,  1,  0, 17, 18,
    8,  8,  8,  8,  7,  6,  5,  4,  3,  2,  1,  0, 17,
    9,  8,  8,  8,  8,  7,  6,  5,  4,  3,  2,  1,  0
  ], .CPU, .C(13 * 13))
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
  let relativePositionBucketGPU = graph.variable(relativePositionBucket.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, causalAttentionMaskGPU, relativePositionBucketGPU)
  reader(state_dict)
  let output = textModel(inputs: tokensTensorGPU, causalAttentionMaskGPU, relativePositionBucketGPU)[0].as(of: Float.self)
  debugPrint(output)
  graph.openStore("/home/liu/workspace/swift-llm/t5_xxl_encoder_f32.ckpt") {
    $0.write("text_model", model: textModel)
  }
}

