import Foundation
import NNC

public enum PythonObject {}

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int)) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
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
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(queries, kOut, vOut).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (Model([x, rot, kIn, vIn], [out, kOut, vOut]), reader)
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
  let kIn = Input()
  let vIn = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm")
  var out = norm1(x)
  let (attention, _) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  let tuple = attention(out, rot, kIn, vIn)
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
  return (Model([x, rot, kIn, vIn], [out, kOut, vOut]), reader)
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
    let tuple = layer(out, rot, kIn, vIn)
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
  return Model([tokens, rot] + kvs, [out] + kvOuts)
}

let graph = DynamicGraph()
graph.withNoGrad {
  let transformer = ModelBuilder { (tokenLengths: (cachedTokenLength: Int, tokenLength: Int), _) in
    return Transformer(Float16.self, vocabularySize: 32_000, width: 4_096, tokenLength: tokenLengths.tokenLength, cachedTokenLength: tokenLengths.cachedTokenLength, layers: 32, MLP: 14336, heads: 32, batchSize: 1)
  }
  graph.maxConcurrency = .limit(1)
  transformer.maxConcurrency = .limit(1)
  var kvs = (0..<64).map { _ in graph.variable(.GPU(0), format: .NHWC, shape: [], of: Float16.self) }
  var tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [5], of: Int32.self)
  tokensTensor[0] = 1
  tokensTensor[1] = 851
  tokensTensor[2] = 349
  tokensTensor[3] = 264
  tokensTensor[4] = 1369
  var rotTensor = graph.variable(.CPU, .NHWC(1, 5, 1, 128), of: Float.self)
  for i in 0..<5 {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  var tokensTensorGPU = tokensTensor.toGPU(0)
  var rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
  transformer.compile((cachedTokenLength: 0, tokenLength: 5), inputs: [tokensTensorGPU, rotTensorGPU] + kvs)
  graph.openStore("/home/liu/workspace/swift-llm/mistral_7b_v0.1_f16.ckpt", flags: .readOnly) {
    $0.read("transformer", model: transformer)
  }
  var tuple = transformer((cachedTokenLength: 0, tokenLength: 5), inputs: tokensTensorGPU, [rotTensorGPU] + kvs).map { $0.as(of: Float16.self) }
  kvs = Array(tuple[1..<65])
  var nextToken = tuple[0].toCPU()
  var topV: Float16 = nextToken[4, 0]
  var topK = 0
  for i in 1..<32_000 {
    if nextToken[4, i] > topV {
      topK = i
      topV = nextToken[4, i]
    }
  }
  tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [1], of: Int32.self)
  tokensTensor[0] = Int32(topK)
  rotTensor = graph.variable(.CPU, .NHWC(1, 1, 1, 128), of: Float.self)
  for k in 0..<64 {
      let theta = Double(5) * 1.0 / pow(10_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, 0, 0, k * 2] = Float(costheta)
      rotTensor[0, 0, 0, k * 2 + 1] = Float(sintheta)
  }
  tokensTensorGPU = tokensTensor.toGPU(0)
  rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
  let startTime = Date()
  tuple = transformer((cachedTokenLength: 5, tokenLength: 1), inputs: tokensTensorGPU, [rotTensorGPU] + kvs).map { $0.as(of: Float16.self) }
  print("time \(Date().timeIntervalSince(startTime))")
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
  print("next top k \(topK)")
  
}
