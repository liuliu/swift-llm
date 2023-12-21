import NNC
import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let torch = Python.import("torch")
let mistral_model = Python.import("mistral.model")
let pathlib = Python.import("pathlib")

let transformer = mistral_model.Transformer.from_folder(pathlib.Path("/home/liu/workspace/mistral-src/mistral-7B-v0.1"), max_batch_size: 1, num_pipeline_ranks: 1)

let state_dict = transformer.state_dict()

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(queries, keys, values).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let q_weight = state_dict["\(prefix).attention.wq.weight"].type(torch.float).cpu()
      .numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let k_weight = state_dict["\(prefix).attention.wk.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    let v_weight = state_dict["\(prefix).attention.wv.weight"].type(torch.float).cpu()
      .numpy()
    tovalues.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: v_weight)))
    let proj_weight = state_dict["\(prefix).attention.wo.weight"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_weight)))
  }
  return (Model([x, rot], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model) {
  let x = Input()
  let w1 = Dense(count: intermediateSize)
  let w3 = Dense(count: intermediateSize)
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize)
  out = w2(out)
  return (w1, w2, w3, Model([x], [out]))
}

func TransformerBlock(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "ffn_norm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP)
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader(state_dict)
    let norm1_weight = state_dict["\(prefix).attention_norm.weight"].type(torch.float)
      .cpu().numpy()
    norm1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm1_weight)))
    let norm2_weight = state_dict["\(prefix).ffn_norm.weight"].type(torch.float)
      .cpu().numpy()
    norm2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm2_weight)))
    let w1_weight = state_dict["\(prefix).feed_forward.w1.weight"].type(torch.float).cpu()
      .numpy()
    w1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w1_weight)))
    let w2_weight = state_dict["\(prefix).feed_forward.w2.weight"].type(torch.float).cpu()
      .numpy()
    w2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w2_weight)))
    let w3_weight = state_dict["\(prefix).feed_forward.w3.weight"].type(torch.float).cpu()
      .numpy()
    w3.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w3_weight)))
  }
  return (Model([x, rot], [out]), reader)
}

public func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["tok_embeddings.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vocab)))
  }
  return (Model([tokens], [embedding], name: "embeddings"), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let (embedding, embedReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: heads / 4, b: batchSize, t: tokenLength,
      MLP: MLP)
    out = layer(out, rot)
    readers.append(reader)
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  let output = Dense(count: vocabularySize, noBias: true)
  out = output(out)
  let reader: (PythonObject) -> Void = { state_dict in
    embedReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_weight = state_dict["norm.weight"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_weight)))
    let output_weight = state_dict["output.weight"].type(torch.float).cpu().numpy()
    output.weight.copy(from: try! Tensor<Float16>(from: Tensor<Float>(numpy: output_weight)))
  }
  return (Model([tokens, rot], [out]), reader)
}

let graph = DynamicGraph()
graph.withNoGrad {
  let (transformer, reader) = Transformer(Float16.self, vocabularySize: 32_000, maxLength: 5, width: 4_096, tokenLength: 5, layers: 32, MLP: 14336, heads: 32, batchSize: 1)
  let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [5], of: Int32.self)
  tokensTensor[0] = 1
  tokensTensor[1] = 851
  tokensTensor[2] = 349
  tokensTensor[3] = 264
  tokensTensor[4] = 1369
  let rotTensor = graph.variable(.CPU, .NHWC(1, 5, 1, 128), of: Float.self)
  for i in 0..<5 {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
  transformer.compile(inputs: tokensTensorGPU, rotTensorGPU)
  reader(state_dict)
  let output = transformer(inputs: tokensTensorGPU, rotTensorGPU).map { $0.as(of: Float16.self) }
  debugPrint(output)
  let nextToken = output[0].toCPU()
  var topV: Float16 = nextToken[4, 0]
  var topK = 0
  for i in 1..<32_000 {
    if nextToken[4, i] > topV {
      topK = i
      topV = nextToken[4, i]
    }
  }
  print("top \(topK)")
}
