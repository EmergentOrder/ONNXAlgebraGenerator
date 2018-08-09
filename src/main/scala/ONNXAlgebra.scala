package org.emergentorder.onnx

import freestyle.free._
import freestyle.free.implicits._
import spire.math.Number
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import scala.language.higherKinds

package object example {
  type Tensor[T] = Tuple2[Vector[T], Seq[Int]]
  trait Operator
  type TTensorFloat16 = Tensor[Float16]

  type T1TensorInt64 = Tensor[Long]

  type VTensorInt64 = Tensor[Long]

  type T1TensorInt32 = Tensor[Int]

  type T1TensorDouble = Tensor[Double]

  type VTensorUint32 = Tensor[UInt]

  type T1TensorUint8 = Tensor[UByte]

  type T2TensorDouble = Tensor[Double]

  type TTensorUint64 = Tensor[ULong]

  type T1TensorInt16 = Tensor[Short]

  type T1TensorUint64 = Tensor[ULong]

  type BTensorBool = Tensor[Boolean]

  type TTensorInt32 = Tensor[Int]

  type TTensorUint8 = Tensor[UByte]

  type T2TensorBool = Tensor[Boolean]

  type TindTensorInt64 = Tensor[Long]

  type VTensorDouble = Tensor[Double]

  type BBool =Boolean

  type T1TensorUint16 = Tensor[UShort]

  type IInt32 =Int

  type IInt64 =Long

  type T2TensorInt64 = Tensor[Long]

  type TTensorBool = Tensor[Boolean]

  type TTensorUint16 = Tensor[UShort]

  type T2TensorInt16 = Tensor[Short]

  type T2TensorFloat16 = Tensor[Float16]

  type VTensorInt16 = Tensor[Short]

  type TTensorInt8 = Tensor[Byte]

  type VTensorString = Tensor[String]

  type T1TensorFloat16 = Tensor[Float16]

  type T1TensorInt8 = Tensor[Byte]

  type TTensorFloat = Tensor[Float]

  type T1TensorString = Tensor[String]

  type TTensorUint32 = Tensor[UInt]

  type TTensorString = Tensor[String]

  type T2TensorUint32 = Tensor[UInt]

  type T2TensorFloat = Tensor[Float]

  type ITensorInt64 = Tensor[Long]

  type TTensorDouble = Tensor[Double]

  type VTensorInt8 = Tensor[Byte]

  type VTensorBool = Tensor[Boolean]

  type T2TensorInt8 = Tensor[Byte]

  type T2TensorUint64 = Tensor[ULong]

  type T2TensorInt32 = Tensor[Int]

  type VTensorUint64 = Tensor[ULong]

  type T2TensorUint8 = Tensor[UByte]

  type VTensorInt32 = Tensor[Int]

  type T1TensorFloat = Tensor[Float]

  type TTensorInt16 = Tensor[Short]

  type T2TensorUint16 = Tensor[UShort]

  type VTensorUint16 = Tensor[UShort]

  type T1TensorUint32 = Tensor[UInt]

  type VTensorFloat = Tensor[Float]

  type T1TensorBool = Tensor[Boolean]

  type VTensorUint8 = Tensor[UByte]

  type VTensorFloat16 = Tensor[Float16]

  type TTensorInt64 = Tensor[Long]

  type TindTensorInt32 = Tensor[Int]
  type T = Tensor[Number]
  type T1 = Tensor[Number]
  type V = Tensor[Number]
  type T2 = Tensor[Number]
  type B = Tensor[Number]
  type Tind = Tensor[Number]
  type I = Tensor[Number]
}
@free trait DataSource {
  def inputData: FS[example.T]
  def getParams(name: String): FS[example.T]
  def getAttributes(name: String): FS[example.T]
}
@free trait Sin extends Operator {

  def Sin(input: example.T)
    : FS[(example.T)]

}
@free trait Atan extends Operator {

  def Atan(input: example.T)
    : FS[(example.T)]

}
@free trait Asin extends Operator {

  def Asin(input: example.T)
    : FS[(example.T)]

}
@free trait Acos extends Operator {

  def Acos(input: example.T)
    : FS[(example.T)]

}
@free trait Unsqueeze extends Operator {

  def Unsqueeze(data: example.T,axes : (Seq[String]))
    : FS[(example.T)]

}
@free trait TopK extends Operator {

  def TopK(X: example.T,axis : Option[(String)] = None,k : (String))
    : FS[(example.T, example.I)]

}
@free trait Tile extends Operator {

  def Tile(input: example.T, repeats: example.T1)
    : FS[(example.T)]

}
@free trait ThresholdedRelu extends Operator {

  def ThresholdedRelu(X: example.T,alpha : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Tanh extends Operator {

  def Tanh(input: example.T)
    : FS[(example.T)]

}
@free trait Sum extends Operator {

  def Sum()
    : FS[(example.T)]

}
@free trait Squeeze extends Operator {

  def Squeeze(data: example.T,axes : (Seq[String]))
    : FS[(example.T)]

}
@free trait SpaceToDepth extends Operator {

  def SpaceToDepth(input: example.T,blocksize : (String))
    : FS[(example.T)]

}
@free trait Softmax extends Operator {

  def Softmax(input: example.T,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Slice extends Operator {

  def Slice(data: example.T,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : FS[(example.T)]

}
@free trait Size extends Operator {

  def Size(data: example.T)
    : FS[(example.T1)]

}
@free trait Shape extends Operator {

  def Shape(data: example.T)
    : FS[(example.T1)]

}
@free trait Selu extends Operator {

  def Selu(X: example.T,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Transpose extends Operator {

  def Transpose(data: example.T,perm : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait ScaledTanh extends Operator {

  def ScaledTanh(input: example.T,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Sigmoid extends Operator {

  def Sigmoid(X: example.T)
    : FS[(example.T)]

}
@free trait Scale extends Operator {

  def Scale(input: example.T,scaleAttr : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait ReduceSumSquare extends Operator {

  def ReduceSumSquare(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ReduceSum extends Operator {

  def ReduceSum(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Reshape extends Operator {

  def Reshape(data: example.T, shape: example.Tensor[Long])
    : FS[(example.T)]

}
@free trait ReduceProd extends Operator {

  def ReduceProd(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Tan extends Operator {

  def Tan(input: example.T)
    : FS[(example.T)]

}
@free trait GlobalAveragePool extends Operator {

  def GlobalAveragePool(X: example.T)
    : FS[(example.T)]

}
@free trait ReduceL2 extends Operator {

  def ReduceL2(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization(input: example.T,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait GRU extends Operator {

  def GRU(X: example.T, W: example.T, R: example.T,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : FS[(example.T, example.T)]

}
@free trait GivenTensorFill extends Operator {

  def GivenTensorFill(shapeInput: Option[example.T] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : FS[(example.T)]

}
@free trait Multinomial extends Operator {

  def Multinomial(input: example.T1,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : FS[(example.T2)]

}
@free trait Flatten extends Operator {

  def Flatten(input: example.T,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Exp extends Operator {

  def Exp(input: example.T)
    : FS[(example.T)]

}
@free trait Equal extends Operator {

  def Equal(A: example.T, B: example.T)
    : FS[(example.T1)]

}
@free trait Not extends Operator {

  def Not(X: example.T)
    : FS[(example.T)]

}
@free trait Sqrt extends Operator {

  def Sqrt(X: example.T)
    : FS[(example.T)]

}
@free trait Elu extends Operator {

  def Elu(X: example.T,alpha : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait ReduceMin extends Operator {

  def ReduceMin(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Div extends Operator {

  def Div(A: example.T, B: example.T)
    : FS[(example.T)]

}
@free trait PRelu extends Operator {

  def PRelu(X: example.T, slope: example.T)
    : FS[(example.T)]

}
@free trait DepthToSpace extends Operator {

  def DepthToSpace(input: example.T,blocksize : (String))
    : FS[(example.T)]

}
@free trait GRUUnit extends Operator {

  def GRUUnit(hidden_prev: example.T, gates: example.T, seq_lengths: example.T, t: example.T,drop_states : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ConvTranspose extends Operator {

  def ConvTranspose(X: example.T, W: example.T,B: Option[example.T] = None,auto_pad : Option[(example.Tensor[Number])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait LogSoftmax extends Operator {

  def LogSoftmax(input: example.T,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ReduceLogSum extends Operator {

  def ReduceLogSum(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ReduceMean extends Operator {

  def ReduceMean(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Crop extends Operator {

  def Crop(input: example.T,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait And extends Operator {

  def And(A: example.T, B: example.T)
    : FS[(example.T1)]

}
@free trait ReduceMax extends Operator {

  def ReduceMax(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ArgMax extends Operator {

  def ArgMax(data: example.T,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(example.Tensor[Long])]

}
@free trait LpNormalization extends Operator {

  def LpNormalization(input: example.T,axis : Option[(String)] = None,p : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Loop extends Operator {

  def Loop(M: example.I, cond: example.B,body : (Seq[Float]))
    : FS[(example.V)]

}
@free trait Affine extends Operator {

  def Affine(X: example.T,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait LSTM extends Operator {

  def LSTM(X: example.T, W: example.T, R: example.T,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None, initial_c: Option[example.T] = None, P: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : FS[(example.T, example.T, example.T)]

}
@free trait Softplus extends Operator {

  def Softplus(X: example.T)
    : FS[(example.T)]

}
@free trait RandomNormalLike extends Operator {

  def RandomNormalLike(input: example.T1,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(example.T2)]

}
@free trait ArgMin extends Operator {

  def ArgMin(data: example.T,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(example.Tensor[Long])]

}
@free trait Conv extends Operator {

  def Conv(X: example.T, W: example.T,B: Option[example.T] = None,auto_pad : Option[(example.Tensor[Number])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait Add extends Operator {

  def Add(A: example.T, B: example.T)
    : FS[(example.T)]

}
@free trait Abs extends Operator {

  def Abs(X: example.T)
    : FS[(example.T)]

}
@free trait Split extends Operator {

  def Split(input: example.T,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait BatchNormalization extends Operator {

  def BatchNormalization(X: example.T, scale: example.T, B: example.T, mean: example.T, someVar: example.T,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(example.T, example.T, example.T, example.T, example.T)]

}
@free trait Upsample extends Operator {

  def Upsample(X: example.T,mode : Option[(example.Tensor[Number])] = None,scaleAttrs : (Seq[Int]))
    : FS[(example.T)]

}
@free trait GlobalLpPool extends Operator {

  def GlobalLpPool(X: example.T,p : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait MatMul extends Operator {

  def MatMul(A: example.T, B: example.T)
    : FS[(example.T)]

}
@free trait Sub extends Operator {

  def Sub(A: example.T, B: example.T)
    : FS[(example.T)]

}
@free trait MaxPool extends Operator {

  def MaxPool(X: example.T,auto_pad : Option[(example.Tensor[Number])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait Neg extends Operator {

  def Neg(X: example.T)
    : FS[(example.T)]

}
@free trait Xor extends Operator {

  def Xor(A: example.T, B: example.T)
    : FS[(example.T1)]

}
@free trait Greater extends Operator {

  def Greater(A: example.T, B: example.T)
    : FS[(example.T1)]

}
@free trait Dropout extends Operator {

  def Dropout(data: example.T,ratio : Option[(Int)] = None)
    : FS[(example.T, example.T)]

}
@free trait Cast extends Operator {

  def Cast(input: example.T1,to : (String))
    : FS[(example.T2)]

}
@free trait Gather extends Operator {

  def Gather(data: example.T, indices: example.Tind,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Ceil extends Operator {

  def Ceil(X: example.T)
    : FS[(example.T)]

}
@free trait Concat extends Operator {

  def Concat()
    : FS[(example.T)]

}
@free trait Softsign extends Operator {

  def Softsign(input: example.T)
    : FS[(example.T)]

}
@free trait ConstantFill extends Operator {

  def ConstantFill(input: Option[example.T1] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : FS[(example.T2)]

}
@free trait Hardmax extends Operator {

  def Hardmax(input: example.T,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Identity extends Operator {

  def Identity(input: example.T)
    : FS[(example.T)]

}
@free trait If extends Operator {

  def If(cond: example.B,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : FS[(example.V)]

}
@free trait ImageScaler extends Operator {

  def ImageScaler(input: example.T,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait RandomUniform extends Operator {

  def RandomUniform()
    : FS[(example.T)]

}
@free trait Cos extends Operator {

  def Cos(input: example.T)
    : FS[(example.T)]

}
@free trait Gemm extends Operator {

  def Gemm(A: example.T, B: example.T, C: example.T,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait InstanceNormalization extends Operator {

  def InstanceNormalization(input: example.T, scale: example.T, B: example.T,epsilon : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Relu extends Operator {

  def Relu(X: example.T)
    : FS[(example.T)]

}
@free trait AveragePool extends Operator {

  def AveragePool(X: example.T,auto_pad : Option[(example.Tensor[Number])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait Less extends Operator {

  def Less(A: example.T, B: example.T)
    : FS[(example.T1)]

}
@free trait Log extends Operator {

  def Log(input: example.T)
    : FS[(example.T)]

}
@free trait LoopIndexTensor extends Operator {

  def LoopIndexTensor(T: example.T, loop_idx: example.I,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Floor extends Operator {

  def Floor(X: example.T)
    : FS[(example.T)]

}
@free trait Min extends Operator {

  def Min()
    : FS[(example.T)]

}
@free trait RNN extends Operator {

  def RNN(X: example.T, W: example.T, R: example.T,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None)
    : FS[(example.T, example.T)]

}
@free trait LpPool extends Operator {

  def LpPool(X: example.T,auto_pad : Option[(example.Tensor[Number])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait Max extends Operator {

  def Max()
    : FS[(example.T)]

}
@free trait MaxRoiPool extends Operator {

  def MaxRoiPool(X: example.T, rois: example.T,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait LRN extends Operator {

  def LRN(X: example.T,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : FS[(example.T)]

}
@free trait Mean extends Operator {

  def Mean()
    : FS[(example.T)]

}
@free trait Mul extends Operator {

  def Mul(A: example.T, B: example.T)
    : FS[(example.T)]

}
@free trait GlobalMaxPool extends Operator {

  def GlobalMaxPool(X: example.T)
    : FS[(example.T)]

}
@free trait Pad extends Operator {

  def Pad(data: example.T,mode : Option[(example.Tensor[Number])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Or extends Operator {

  def Or(A: example.T, B: example.T)
    : FS[(example.T1)]

}
@free trait ReduceL1 extends Operator {

  def ReduceL1(data: example.T,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ParametricSoftplus extends Operator {

  def ParametricSoftplus(X: example.T,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Pow extends Operator {

  def Pow(X: example.T, Y: example.T)
    : FS[(example.T)]

}
@free trait Constant extends Operator {

  def Constant()
    : FS[(example.T)]

}
@free trait RandomNormal extends Operator {

  def RandomNormal()
    : FS[(example.T)]

}
@free trait HardSigmoid extends Operator {

  def HardSigmoid(X: example.T,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Clip extends Operator {

  def Clip(input: example.T,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait RandomUniformLike extends Operator {

  def RandomUniformLike(input: example.T1,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(example.T2)]

}
@free trait LeakyRelu extends Operator {

  def LeakyRelu(X: example.T,alpha : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Reciprocal extends Operator {

  def Reciprocal(X: example.T)
    : FS[(example.T)]

}