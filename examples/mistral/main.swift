import Foundation
import NNC
import SentencePiece

public enum PythonObject {}

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int)) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  let kIn = Input()
  let vIn = Input()
  var keys = tokeys(x).reshaped([b, t.1, hk, k])
  var queries = toqueries(x).reshaped([b, t.1, h, k])
  let values = tovalues(x).reshaped([b, t.1, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  let kOut = Functional.concat(axis: 1, kIn, keys)
  let vOut = Functional.concat(axis: 1, vIn, values)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true, hasAttentionMask: true)(queries, kOut, vOut, causalAttentionMask).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (Model([x, rot, causalAttentionMask, kIn, vIn], [out, kOut, vOut]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (Model, Model, Model, Model) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true)
  let w3 = Dense(count: intermediateSize, noBias: true)
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true)
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

func TransformerBlock(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int), MLP: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let kIn = Input()
  let vIn = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm")
  var out = norm1(x)
  let (attention, _) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  let tuple = attention(out, rot, causalAttentionMask, kIn, vIn)
  out = tuple[0] + x
  let kOut = tuple[1]
  let vOut = tuple[2]
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "ffn_norm")
  out = norm2(out)
  let (_, _, _, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "ffn")
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (Model([x, rot, causalAttentionMask, kIn, vIn], [out, kOut, vOut]), reader)
}

public func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (Model([tokens], [embedding]), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: Int, cachedTokenLength: Int, layers: Int, MLP: Int, heads: Int, batchSize: Int
) -> Model {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  var kvs = [Input]()
  var kvOuts = [Model.IO]()
  let (embedding, _) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, embeddingSize: width)
  var out = embedding(tokens)
  for i in 0..<layers {
    let (layer, _) = TransformerBlock(
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: heads / 4, b: batchSize, t: (cachedTokenLength + tokenLength, tokenLength),
      MLP: MLP)
    let kIn = Input()
    let vIn = Input()
    let tuple = layer(out, rot, causalAttentionMask, kIn, vIn)
    out = tuple[0]
    kvs.append(kIn)
    kvs.append(vIn)
    kvOuts.append(tuple[1])
    kvOuts.append(tuple[2])
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  let output = Dense(count: vocabularySize, noBias: true, name: "output")
  out = output(out)
  return Model([tokens, rot, causalAttentionMask] + kvs, [out] + kvOuts)
}

let graph = DynamicGraph()
graph.withNoGrad {
  let transformer = ModelBuilder { (tokenLengths: (cachedTokenLength: Int, tokenLength: Int), _) in
    return Transformer(Float16.self, vocabularySize: 32_000, width: 4_096, tokenLength: tokenLengths.tokenLength, cachedTokenLength: tokenLengths.cachedTokenLength, layers: 32, MLP: 14336, heads: 32, batchSize: 1)
  }
  graph.maxConcurrency = .limit(1)
  transformer.maxConcurrency = .limit(1)
  var kvs = (0..<64).map { _ in graph.variable(.GPU(0), format: .NHWC, shape: [], of: Float16.self) }
  let sentencePiece = SentencePiece(file: "/Users/liu/workspace/swift-llm/examples/mistral/tokenizer.model")
  var ids = sentencePiece.encode("This is a test").map { $0.id }
  var tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [ids.count + 1], of: Int32.self)
  tokensTensor[0] = 1
  for i in 0..<ids.count {
    tokensTensor[i + 1] = ids[i]
  }
  var rotTensor = graph.variable(.CPU, .NHWC(1, ids.count + 1, 1, 128), of: Float.self)
  for i in 0..<(ids.count + 1) {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  var causalAttentionMask = graph.variable(
    .CPU, .NHWC(1, 1, ids.count + 1, ids.count + 1), of: Float16.self)
  causalAttentionMask.full(0)
  for i in 0..<ids.count {
    for j in (i + 1)..<(ids.count + 1) {
      causalAttentionMask[0, 0, i, j] = -Float16.greatestFiniteMagnitude
    }
  }
  var causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
  var tokensTensorGPU = tokensTensor.toGPU(0)
  var rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
  transformer.compile((cachedTokenLength: 0, tokenLength: ids.count + 1), inputs: [tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU] + kvs)
  graph.openStore("/Users/liu/workspace/swift-llm/mistral_7b_v0.1_f16.ckpt", flags: .readOnly) {
    $0.read("transformer", model: transformer)
  }
  var tuple = transformer((cachedTokenLength: 0, tokenLength: ids.count + 1), inputs: tokensTensorGPU, [rotTensorGPU, causalAttentionMaskGPU] + kvs).map { $0.as(of: Float16.self) }
  kvs = Array(tuple[1..<65])
  let kvs100 = kvs.map {
    let v = graph.variable(.GPU(0), .NHWC($0.shape[0], 64, $0.shape[2], $0.shape[3]), of: Float16.self)
    v.full(0)
    return v
  }
  var nextToken = tuple[0].toCPU()
  debugPrint(nextToken)
  var topV: Float16 = nextToken[ids.count, 0]
  var topK = 0
  for i in 1..<32_000 {
    if nextToken[ids.count, i] > topV {
      topK = i
      topV = nextToken[ids.count, i]
    }
  }
  ids.append(Int32(topK))
  causalAttentionMask = graph.variable(
    .CPU, .NHWC(1, 1, 1, 65), of: Float16.self)
  causalAttentionMask.full(0)
  causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
  transformer.compile((cachedTokenLength: 64, tokenLength: 1), inputs: [tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU] + kvs100, isEager: true)
  let startTime = Date()
  let maxTokens = 35
  for _ in 0..<maxTokens {
    tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [1], of: Int32.self)
    tokensTensor[0] = Int32(topK)
    rotTensor = graph.variable(.CPU, .NHWC(1, 1, 1, 128), of: Float.self)
    let cachedTokenLength = ids.count
    for k in 0..<64 {
        let theta = Double(cachedTokenLength) * 1.0 / pow(10_000, Double(k) * 2 / 128)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, 0, 0, k * 2] = Float(costheta)
        rotTensor[0, 0, 0, k * 2 + 1] = Float(sintheta)
    }
    causalAttentionMask = graph.variable(
      .CPU, .NHWC(1, 1, 1, ids.count + 1), of: Float16.self)
    causalAttentionMask.full(0)
    causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
    tokensTensorGPU = tokensTensor.toGPU(0)
    rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
    tuple = transformer((cachedTokenLength: cachedTokenLength, tokenLength: 1), inputs: tokensTensorGPU, [rotTensorGPU, causalAttentionMaskGPU] + kvs).map { $0.as(of: Float16.self) }
    kvs = Array(tuple[1..<65])
    nextToken = tuple[0].toCPU()
    topV = nextToken[0, 0]
    topK = 0
    for i in 1..<32_000 {
      if nextToken[0, i] > topV {
        topK = i
        topV = nextToken[0, i]
      }
    }
    ids.append(Int32(topK))
    print(sentencePiece.decode(ids))
  }
  print("\(Double(maxTokens) / Date().timeIntervalSince(startTime)) tok/s")
  print(sentencePiece.decode(ids))
}
