package example

import freestyle.free._
import freestyle.free.implicits._
import spire.math.Number
import scala.language.higherKinds

package object example {
  type Tensor[T] = Tuple2[Vector[T], Seq[Int]]
  trait Operator
  type T1Float16 = Tensor[Float16]

  type T1Int64 = Tensor[Long]

  type TInt64 = Tensor[Long]

  type T2Int32 = Tensor[Int]

  type T2Float = Tensor[Float]

  type TInt32 = Tensor[Int]

  type T2Double = Tensor[Double]

  type T2Int64 = Tensor[Long]

  type T2Bool = Tensor[Boolean]

  type T1Double = Tensor[Double]

  type TindInt64 = Tensor[Long]

  type TDouble = Tensor[Double]

  type TFloat = Tensor[Float]

  type TBool = Tensor[Boolean]

  type TindInt32 = Tensor[Int]

  type T1Bool = Tensor[Boolean]

  type T1Float = Tensor[Float]

  type T1Int32 = Tensor[Int]

  type TFloat16 = Tensor[Float16]

  type T2Float16 = Tensor[Float16]
  type T1 = Tensor[Number]
  type T = Tensor[Number]
  type T2 = Tensor[Number]
  type Tind = Tensor[Number]
}
@free trait DataSource {
  def inputData: FS[example.T]
  def getParams(name: String): FS[example.T]
  def getAttributes(name: String): FS[example.T]
}
@free trait LSTM extends Operator {

  def LSTM(X: example.T,
           W: example.T,
           R: example.T,
           B: Option[example.T] = None,
           sequence_lens: Option[example.T1] = None,
           initial_h: Option[example.T] = None,
           initial_c: Option[example.T] = None,
           P: Option[example.T] = None,
           activation_alpha: Option[(Seq[Float])] = None,
           activation_beta: Option[(Seq[Float])] = None,
           activations: Option[(Seq[String])] = None,
           clip: Option[(Float)] = None,
           direction: Option[(String)] = None,
           hidden_size: Option[(Int)] = None,
           input_forget: Option[(Int)] = None,
           output_sequence: Option[(Int)] = None): FS[(example.T, example.T)]

}
@free trait GRU extends Operator {

  def GRU(X: example.T,
          W: example.T,
          R: example.T,
          B: Option[example.T] = None,
          sequence_lens: Option[example.T1] = None,
          initial_h: Option[example.T] = None,
          activation_alpha: Option[(Seq[Float])] = None,
          activation_beta: Option[(Seq[Float])] = None,
          activations: Option[(Seq[String])] = None,
          clip: Option[(Float)] = None,
          direction: Option[(String)] = None,
          hidden_size: Option[(Int)] = None,
          output_sequence: Option[(Int)] = None): FS[(example.T, example.T)]

}
@free trait RNN extends Operator {

  def RNN(X: example.T,
          W: example.T,
          R: example.T,
          B: Option[example.T] = None,
          sequence_lens: Option[example.T1] = None,
          initial_h: Option[example.T] = None,
          activation_alpha: Option[(Seq[Float])] = None,
          activation_beta: Option[(Seq[Float])] = None,
          activations: Option[(Seq[String])] = None,
          clip: Option[(Float)] = None,
          direction: Option[(String)] = None,
          hidden_size: Option[(Int)] = None,
          output_sequence: Option[(Int)] = None): FS[(example.T, example.T)]

}
@free trait Not extends Operator {

  def Not(X: example.T): FS[(example.T)]

}
@free trait Equal extends Operator {

  def Equal(A: example.T,
            B: example.T,
            axis: Option[(Int)] = None,
            broadcast: Option[(Int)] = None): FS[(example.T1)]

}
@free trait Or extends Operator {

  def Or(A: example.T,
         B: example.T,
         axis: Option[(Int)] = None,
         broadcast: Option[(Int)] = None): FS[(example.T1)]

}
@free trait And extends Operator {

  def And(A: example.T,
          B: example.T,
          axis: Option[(Int)] = None,
          broadcast: Option[(Int)] = None): FS[(example.T1)]

}
@free trait ArgMin extends Operator {

  def ArgMin(data: example.T,
             axis: Option[(Int)] = None,
             keepdims: Option[(Int)] = None): FS[(example.Tensor[Int])]

}
@free trait ArgMax extends Operator {

  def ArgMax(data: example.T,
             axis: Option[(Int)] = None,
             keepdims: Option[(Int)] = None): FS[(example.Tensor[Int])]

}
@free trait ReduceL2 extends Operator {

  def ReduceL2(data: example.T,
               axes: Option[(Seq[Int])] = None,
               keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait ReduceLogSum extends Operator {

  def ReduceLogSum(data: example.T,
                   axes: Option[(Seq[Int])] = None,
                   keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait ReduceSumSquare extends Operator {

  def ReduceSumSquare(data: example.T,
                      axes: Option[(Seq[Int])] = None,
                      keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait ReduceSum extends Operator {

  def ReduceSum(data: example.T,
                axes: Option[(Seq[Int])] = None,
                keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait ReduceMax extends Operator {

  def ReduceMax(data: example.T,
                axes: Option[(Seq[Int])] = None,
                keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait Flatten extends Operator {

  def Flatten(input: example.T, axis: Option[(Int)] = None): FS[(example.T)]

}
@free trait Exp extends Operator {

  def Exp(input: example.T): FS[(example.T)]

}
@free trait Pow extends Operator {

  def Pow(X: example.T, Y: example.T): FS[(example.T)]

}
@free trait GivenTensorFill extends Operator {

  def GivenTensorFill(shapeInput: Option[example.T] = None,
                      extra_shape: Option[(Seq[Int])] = None,
                      input_as_shape: Option[(Int)] = None,
                      shape: Option[(Seq[Int])] = None,
                      values: Option[(Seq[Float])] = None): FS[(example.T)]

}
@free trait Abs extends Operator {

  def Abs(X: example.T): FS[(example.T)]

}
@free trait Selu extends Operator {

  def Selu(X: example.T,
           alpha: Option[(Float)] = None,
           gamma: Option[(Float)] = None): FS[(example.T)]

}
@free trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp(data: example.T,
                      axes: Option[(Seq[Int])] = None,
                      keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait MatMul extends Operator {

  def MatMul(A: example.T, B: example.T): FS[(example.T)]

}
@free trait Identity extends Operator {

  def Identity(input: example.T): FS[(example.T)]

}
@free trait ReduceMin extends Operator {

  def ReduceMin(data: example.T,
                axes: Option[(Seq[Int])] = None,
                keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait Div extends Operator {

  def Div(A: example.T,
          B: example.T,
          axis: Option[(Int)] = None,
          broadcast: Option[(Int)] = None): FS[(example.T)]

}
@free trait PRelu extends Operator {

  def PRelu(X: example.T, slope: example.T): FS[(example.T)]

}
@free trait LpNormalization extends Operator {

  def LpNormalization(input: example.T,
                      axis: Option[(Int)] = None,
                      p: Option[(Int)] = None): FS[(example.T)]

}
@free trait Sum extends Operator {

  def Sum(): FS[(example.T)]

}
@free trait Concat extends Operator {

  def Concat(): FS[(example.T)]

}
@free trait Cast extends Operator {

  def Cast(input: example.T1, to: Option[(String)] = None): FS[(example.T2)]

}
@free trait Slice extends Operator {

  def Slice(data: example.T,
            axes: Option[(Seq[Int])] = None,
            ends: (Seq[Int]),
            starts: (Seq[Int])): FS[(example.T)]

}
@free trait Mul extends Operator {

  def Mul(A: example.T,
          B: example.T,
          axis: Option[(Int)] = None,
          broadcast: Option[(Int)] = None): FS[(example.T)]

}
@free trait ReduceProd extends Operator {

  def ReduceProd(data: example.T,
                 axes: Option[(Seq[Int])] = None,
                 keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait Reshape extends Operator {

  def Reshape(data: example.T,
              shape: Option[(Seq[Int])] = None): FS[(example.T)]

}
@free trait Ceil extends Operator {

  def Ceil(X: example.T): FS[(example.T)]

}
@free trait Gather extends Operator {

  def Gather(data: example.T,
             indices: example.Tind,
             axis: Option[(Int)] = None): FS[(example.T)]

}
@free trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization(
      input: example.T,
      across_channels: Option[(Int)] = None,
      normalize_variance: Option[(Int)] = None): FS[(example.T)]

}
@free trait Tanh extends Operator {

  def Tanh(input: example.T): FS[(example.T)]

}
@free trait SpaceToDepth extends Operator {

  def SpaceToDepth(input: example.T,
                   blocksize: Option[(Int)] = None): FS[(example.T)]

}
@free trait Conv extends Operator {

  def Conv(X: example.T,
           W: example.T,
           B: Option[example.T] = None,
           auto_pad: Option[(String)] = None,
           dilations: Option[(Seq[Int])] = None,
           group: Option[(Int)] = None,
           kernel_shape: Option[(Seq[Int])] = None,
           pads: Option[(Seq[Int])] = None,
           strides: Option[(Seq[Int])] = None): FS[(example.T)]

}
@free trait Add extends Operator {

  def Add(A: example.T,
          B: example.T,
          axis: Option[(Int)] = None,
          broadcast: Option[(Int)] = None): FS[(example.T)]

}
@free trait Neg extends Operator {

  def Neg(X: example.T): FS[(example.T)]

}
@free trait Sub extends Operator {

  def Sub(A: example.T,
          B: example.T,
          axis: Option[(Int)] = None,
          broadcast: Option[(Int)] = None): FS[(example.T)]

}
@free trait MaxPool extends Operator {

  def MaxPool(X: example.T,
              auto_pad: Option[(String)] = None,
              kernel_shape: Option[(Seq[Int])] = None,
              pads: Option[(Seq[Int])] = None,
              strides: Option[(Seq[Int])] = None): FS[(example.T)]

}
@free trait Squeeze extends Operator {

  def Squeeze(data: example.T, axes: (Seq[Int])): FS[(example.T)]

}
@free trait ReduceMean extends Operator {

  def ReduceMean(data: example.T,
                 axes: Option[(Seq[Int])] = None,
                 keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait Crop extends Operator {

  def Crop(input: example.T,
           border: Option[(Seq[Int])] = None,
           scaleAttr: Option[(Seq[Int])] = None): FS[(example.T)]

}
@free trait Elu extends Operator {

  def Elu(X: example.T, alpha: Option[(Float)] = None): FS[(example.T)]

}
@free trait Sqrt extends Operator {

  def Sqrt(X: example.T): FS[(example.T)]

}
@free trait DepthToSpace extends Operator {

  def DepthToSpace(input: example.T,
                   blocksize: Option[(Int)] = None): FS[(example.T)]

}
@free trait GRUUnit extends Operator {

  def GRUUnit(hidden_prev: example.T,
              gates: example.T,
              seq_lengths: example.T,
              t: example.T,
              drop_states: Option[(Int)] = None): FS[(example.T)]

}
@free trait LeakyRelu extends Operator {

  def LeakyRelu(X: example.T, alpha: Option[(Float)] = None): FS[(example.T)]

}
@free trait Reciprocal extends Operator {

  def Reciprocal(X: example.T): FS[(example.T)]

}
@free trait RandomUniform extends Operator {

  def RandomUniform(): FS[(example.T)]

}
@free trait ImageScaler extends Operator {

  def ImageScaler(input: example.T,
                  bias: Option[(Seq[Float])] = None,
                  scaleAttr: Option[(Float)] = None): FS[(example.T)]

}
@free trait Tile extends Operator {

  def Tile(input: example.T, tiles: example.T, axis: example.T): FS[(example.T)]

}
@free trait Constant extends Operator {

  def Constant(): FS[(example.T)]

}
@free trait RandomNormal extends Operator {

  def RandomNormal(): FS[(example.T)]

}
@free trait Log extends Operator {

  def Log(input: example.T): FS[(example.T)]

}
@free trait Clip extends Operator {

  def Clip(input: example.T,
           max: Option[(Float)] = None,
           min: Option[(Float)] = None): FS[(example.T)]

}
@free trait RandomUniformLike extends Operator {

  def RandomUniformLike(input: example.T,
                        dtype: Option[(Int)] = None,
                        high: Option[(Float)] = None,
                        low: Option[(Float)] = None,
                        seed: Option[(Float)] = None): FS[(example.T)]

}
@free trait HardSigmoid extends Operator {

  def HardSigmoid(X: example.T,
                  alpha: Option[(Float)] = None,
                  beta: Option[(Float)] = None): FS[(example.T)]

}
@free trait MaxRoiPool extends Operator {

  def MaxRoiPool(X: example.T,
                 rois: example.T,
                 pooled_shape: Option[(Seq[Int])] = None,
                 spatial_scaleAttr: Option[(Float)] = None): FS[(example.T)]

}
@free trait Floor extends Operator {

  def Floor(X: example.T): FS[(example.T)]

}
@free trait Min extends Operator {

  def Min(): FS[(example.T)]

}
@free trait GlobalLpPool extends Operator {

  def GlobalLpPool(X: example.T, p: Option[(Int)] = None): FS[(example.T)]

}
@free trait Upsample extends Operator {

  def Upsample(X: example.T,
               height_scaleAttr: (Float),
               mode: Option[(String)] = None,
               width_scaleAttr: (Float)): FS[(example.T)]

}
@free trait LRN extends Operator {

  def LRN(X: example.T,
          alpha: (Float),
          beta: (Float),
          bias: Option[(Float)] = None,
          size: (Int)): FS[(example.T)]

}
@free trait Mean extends Operator {

  def Mean(): FS[(example.T)]

}
@free trait ThresholdedRelu extends Operator {

  def ThresholdedRelu(X: example.T,
                      alpha: Option[(Float)] = None): FS[(example.T)]

}
@free trait Softmax extends Operator {

  def Softmax(input: example.T, axis: Option[(Int)] = None): FS[(example.T)]

}
@free trait GlobalAveragePool extends Operator {

  def GlobalAveragePool(X: example.T): FS[(example.T)]

}
@free trait LogSoftmax extends Operator {

  def LogSoftmax(input: example.T, axis: Option[(Int)] = None): FS[(example.T)]

}
@free trait ConvTranspose extends Operator {

  def ConvTranspose(X: example.T,
                    W: example.T,
                    B: Option[example.T] = None,
                    auto_pad: Option[(String)] = None,
                    dilations: Option[(Seq[Int])] = None,
                    group: Option[(Int)] = None,
                    kernel_shape: Option[(Seq[Int])] = None,
                    output_shape: Option[(Seq[Int])] = None,
                    pads: Option[(Seq[Int])] = None,
                    strides: Option[(Seq[Int])] = None): FS[(example.T)]

}
@free trait Hardmax extends Operator {

  def Hardmax(input: example.T, axis: Option[(Int)] = None): FS[(example.T)]

}
@free trait RandomNormalLike extends Operator {

  def RandomNormalLike(input: example.T,
                       dtype: Option[(Int)] = None,
                       mean: Option[(Float)] = None,
                       scaleAttr: Option[(Float)] = None,
                       seed: Option[(Float)] = None): FS[(example.T)]

}
@free trait Softplus extends Operator {

  def Softplus(X: example.T): FS[(example.T)]

}
@free trait Affine extends Operator {

  def Affine(X: example.T,
             alpha: Option[(Float)] = None,
             beta: Option[(Float)] = None): FS[(example.T)]

}
@free trait Transpose extends Operator {

  def Transpose(data: example.T,
                perm: Option[(Seq[Int])] = None): FS[(example.T)]

}
@free trait ScaledTanh extends Operator {

  def ScaledTanh(input: example.T,
                 alpha: Option[(Float)] = None,
                 beta: Option[(Float)] = None): FS[(example.T)]

}
@free trait Less extends Operator {

  def Less(A: example.T,
           B: example.T,
           axis: Option[(Int)] = None,
           broadcast: Option[(Int)] = None): FS[(example.T1)]

}
@free trait Relu extends Operator {

  def Relu(X: example.T): FS[(example.T)]

}
@free trait AveragePool extends Operator {

  def AveragePool(X: example.T,
                  auto_pad: Option[(String)] = None,
                  kernel_shape: Option[(Seq[Int])] = None,
                  pads: Option[(Seq[Int])] = None,
                  strides: Option[(Seq[Int])] = None): FS[(example.T)]

}
@free trait ReduceL1 extends Operator {

  def ReduceL1(data: example.T,
               axes: Option[(Seq[Int])] = None,
               keepdims: Option[(Int)] = None): FS[(example.T)]

}
@free trait ParametricSoftplus extends Operator {

  def ParametricSoftplus(X: example.T,
                         alpha: Option[(Float)] = None,
                         beta: Option[(Float)] = None): FS[(example.T)]

}
@free trait Softsign extends Operator {

  def Softsign(input: example.T): FS[(example.T)]

}
@free trait ConstantFill extends Operator {

  def ConstantFill(input: Option[example.T1] = None,
                   dtype: Option[(Int)] = None,
                   extra_shape: Option[(Seq[Int])] = None,
                   input_as_shape: Option[(Int)] = None,
                   shape: Option[(Seq[Int])] = None,
                   value: Option[(Float)] = None): FS[(example.T2)]

}
@free trait Max extends Operator {

  def Max(): FS[(example.T)]

}
@free trait FC extends Operator {

  def FC(X: example.T,
         W: example.T,
         B: example.T,
         axis: Option[(Int)] = None,
         axis_w: Option[(Int)] = None): FS[(example.T)]

}
@free trait Sigmoid extends Operator {

  def Sigmoid(X: example.T): FS[(example.T)]

}
@free trait Scale extends Operator {

  def Scale(input: example.T,
            scaleAttr: Option[(Float)] = None): FS[(example.T)]

}
@free trait Greater extends Operator {

  def Greater(A: example.T,
              B: example.T,
              axis: Option[(Int)] = None,
              broadcast: Option[(Int)] = None): FS[(example.T1)]

}
@free trait Xor extends Operator {

  def Xor(A: example.T,
          B: example.T,
          axis: Option[(Int)] = None,
          broadcast: Option[(Int)] = None): FS[(example.T1)]

}
@free trait Dropout extends Operator {

  def Dropout(data: example.T,
              is_test: Option[(Int)] = None,
              ratio: Option[(Float)] = None): FS[(example.T, example.T)]

}
@free trait Embedding extends Operator {

  def Embedding(
      input: example.Tensor[Long],
      input_dim: Option[(Int)] = None,
      output_dim: Option[(Int)] = None,
      weights: Option[(example.Tensor[Number])] = None): FS[(example.T)]

}
@free trait LpPool extends Operator {

  def LpPool(X: example.T,
             auto_pad: Option[(String)] = None,
             kernel_shape: Option[(Seq[Int])] = None,
             p: Option[(Int)] = None,
             pads: Option[(Seq[Int])] = None,
             strides: Option[(Seq[Int])] = None): FS[(example.T)]

}
@free trait Pad extends Operator {

  def Pad(data: example.T,
          mode: Option[(String)] = None,
          pads: (Seq[Int]),
          value: Option[(Float)] = None): FS[(example.T)]

}
@free trait GlobalMaxPool extends Operator {

  def GlobalMaxPool(X: example.T): FS[(example.T)]

}
@free trait Split extends Operator {

  def Split(input: example.T,
            axis: Option[(Int)] = None,
            splitAttr: Option[(Seq[Int])] = None): FS[(example.T)]

}
@free trait BatchNormalization extends Operator {

  def BatchNormalization(X: example.T,
                         scale: example.T,
                         B: example.T,
                         mean: example.T,
                         someVar: example.T,
                         epsilon: Option[(Float)] = None,
                         is_test: Option[(Int)] = None,
                         momentum: Option[(Float)] = None,
                         spatial: Option[(Int)] = None)
    : FS[(example.T, example.T, example.T, example.T, example.T)]

}
@free trait Gemm extends Operator {

  def Gemm(A: example.T,
           B: example.T,
           C: example.T,
           alpha: Option[(Float)] = None,
           beta: Option[(Float)] = None,
           broadcast: Option[(Int)] = None,
           transA: Option[(Int)] = None,
           transB: Option[(Int)] = None): FS[(example.T)]

}
@free trait InstanceNormalization extends Operator {

  def InstanceNormalization(input: example.T,
                            scale: example.T,
                            B: example.T,
                            epsilon: Option[(Float)] = None): FS[(example.T)]

}
