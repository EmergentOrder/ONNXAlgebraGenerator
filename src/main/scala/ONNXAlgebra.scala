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
@free trait GlobalMaxPool extends Operator {

  def GlobalMaxPool1(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait HardSigmoid extends Operator {

  def HardSigmoid1(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def HardSigmoid6(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait LeakyRelu extends Operator {

  def LeakyRelu1(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def LeakyRelu6(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait TopK extends Operator {

  def TopK1(name: String,X: example.T, Xname: String,axis : Option[(String)] = None,k : (String))
    : FS[(example.T, example.I)]

}
@free trait ReduceSumSquare extends Operator {

  def ReduceSumSquare1(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Hardmax extends Operator {

  def Hardmax1(name: String,input: example.T, inputname: String,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait MatMul extends Operator {

  def MatMul1(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T)]

}
@free trait Sum extends Operator {

  def Sum1(name: String)
    : FS[(example.T)]


  def Sum6(name: String)
    : FS[(example.T)]

}
@free trait Tanh extends Operator {

  def Tanh1(name: String,input: example.T, inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Tanh6(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait Less extends Operator {

  def Less1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T1)]


  def Less7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T1)]

}
@free trait Log extends Operator {

  def Log1(name: String,input: example.T, inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Log6(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait GRUUnit extends Operator {

  def GRUUnit1(name: String,hidden_prev: example.T, hidden_prevname: String, gates: example.T, gatesname: String, seq_lengths: example.T, seq_lengthsname: String, t: example.T, tname: String,drop_states : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Sin extends Operator {

  def Sin7(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait Acos extends Operator {

  def Acos7(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait Abs extends Operator {

  def Abs1(name: String,X: example.T, Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Abs6(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait PRelu extends Operator {

  def PRelu1(name: String,X: example.T, Xname: String, slope: example.T, slopename: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def PRelu6(name: String,X: example.T, Xname: String, slope: example.T, slopename: String)
    : FS[(example.T)]


  def PRelu7(name: String,X: example.T, Xname: String, slope: example.T, slopename: String)
    : FS[(example.T)]

}
@free trait Softplus extends Operator {

  def Softplus1(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait MaxRoiPool extends Operator {

  def MaxRoiPool1(name: String,X: example.T, Xname: String, rois: example.T, roisname: String,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait GivenTensorFill extends Operator {

  def GivenTensorFill1(name: String,shapeInput: Option[example.T] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : FS[(example.T)]

}
@free trait Shape extends Operator {

  def Shape1(name: String,data: example.T, dataname: String)
    : FS[(example.T1)]

}
@free trait InstanceNormalization extends Operator {

  def InstanceNormalization1(name: String,input: example.T, inputname: String, scale: example.T, scalename: String, B: example.T, Bname: String,consumed_inputs : Option[(Seq[String])] = None,epsilon : Option[(Int)] = None)
    : FS[(example.T)]


  def InstanceNormalization6(name: String,input: example.T, inputname: String, scale: example.T, scalename: String, B: example.T, Bname: String,epsilon : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Pad extends Operator {

  def Pad1(name: String,data: example.T, dataname: String,mode : Option[(example.Tensor[Number])] = None,paddings : (Seq[String]),value : Option[(Int)] = None)
    : FS[(example.T)]


  def Pad2(name: String,data: example.T, dataname: String,mode : Option[(example.Tensor[Number])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Squeeze extends Operator {

  def Squeeze1(name: String,data: example.T, dataname: String,axes : (Seq[String]))
    : FS[(example.T)]

}
@free trait Identity extends Operator {

  def Identity1(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait Softmax extends Operator {

  def Softmax1(name: String,input: example.T, inputname: String,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ReduceMax extends Operator {

  def ReduceMax1(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Gather extends Operator {

  def Gather1(name: String,data: example.T, dataname: String, indices: example.Tind, indicesname: String,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Min extends Operator {

  def Min1(name: String)
    : FS[(example.T)]


  def Min6(name: String)
    : FS[(example.T)]

}
@free trait Div extends Operator {

  def Div1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Div6(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T)]


  def Div7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T)]

}
@free trait Not extends Operator {

  def Not1(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait ArgMin extends Operator {

  def ArgMin1(name: String,data: example.T, dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(example.Tensor[Long])]

}
@free trait Transpose extends Operator {

  def Transpose1(name: String,data: example.T, dataname: String,perm : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait Equal extends Operator {

  def Equal1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T1)]


  def Equal7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T1)]

}
@free trait Constant extends Operator {

  def Constant1(name: String)
    : FS[(example.T)]

}
@free trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization1(name: String,input: example.T, inputname: String,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait And extends Operator {

  def And1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T1)]


  def And7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T1)]

}
@free trait RandomUniformLike extends Operator {

  def RandomUniformLike1(name: String,input: example.T1, inputname: String,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(example.T2)]

}
@free trait ReduceMean extends Operator {

  def ReduceMean1(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Gemm extends Operator {

  def Gemm1(name: String,A: example.T, Aname: String, B: example.T, Bname: String, C: example.T, Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(example.T)]


  def Gemm6(name: String,A: example.T, Aname: String, B: example.T, Bname: String, C: example.T, Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(example.T)]


  def Gemm7(name: String,A: example.T, Aname: String, B: example.T, Bname: String, C: example.T, Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Tile extends Operator {

  def Tile1(name: String,input: example.T, inputname: String, tiles: example.T, tilesname: String, axis: example.T, axisname: String)
    : FS[(example.T)]


  def Tile6(name: String,input: example.T, inputname: String, repeats: example.T1, repeatsname: String)
    : FS[(example.T)]

}
@free trait LpNormalization extends Operator {

  def LpNormalization1(name: String,input: example.T, inputname: String,axis : Option[(String)] = None,p : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ReduceProd extends Operator {

  def ReduceProd1(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Scale extends Operator {

  def Scale1(name: String,input: example.T, inputname: String,scaleAttr : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Flatten extends Operator {

  def Flatten1(name: String,input: example.T, inputname: String,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Dropout extends Operator {

  def Dropout1(name: String,data: example.T, dataname: String,consumed_inputs : Option[(Seq[String])] = None,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(example.T, example.T)]


  def Dropout6(name: String,data: example.T, dataname: String,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(example.T, example.T)]


  def Dropout7(name: String,data: example.T, dataname: String,ratio : Option[(Int)] = None)
    : FS[(example.T, example.T)]

}
@free trait Multinomial extends Operator {

  def Multinomial7(name: String,input: example.T1, inputname: String,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : FS[(example.T2)]

}
@free trait AveragePool extends Operator {

  def AveragePool1(name: String,X: example.T, Xname: String,auto_pad : Option[(example.Tensor[Number])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def AveragePool7(name: String,X: example.T, Xname: String,auto_pad : Option[(example.Tensor[Number])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait RandomNormalLike extends Operator {

  def RandomNormalLike1(name: String,input: example.T1, inputname: String,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(example.T2)]

}
@free trait ReduceL1 extends Operator {

  def ReduceL11(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ThresholdedRelu extends Operator {

  def ThresholdedRelu1(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait LpPool extends Operator {

  def LpPool1(name: String,X: example.T, Xname: String,auto_pad : Option[(example.Tensor[Number])] = None,kernel_shape : Option[(Seq[String])] = None,p : Option[(Int)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def LpPool2(name: String,X: example.T, Xname: String,auto_pad : Option[(example.Tensor[Number])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait Tan extends Operator {

  def Tan7(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait Reciprocal extends Operator {

  def Reciprocal1(name: String,X: example.T, Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Reciprocal6(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait Elu extends Operator {

  def Elu1(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Elu6(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Size extends Operator {

  def Size1(name: String,data: example.T, dataname: String)
    : FS[(example.T1)]

}
@free trait Concat extends Operator {

  def Concat1(name: String)
    : FS[(example.T)]


  def Concat4(name: String)
    : FS[(example.T)]

}
@free trait ScaledTanh extends Operator {

  def ScaledTanh1(name: String,input: example.T, inputname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait ReduceSum extends Operator {

  def ReduceSum1(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait GlobalAveragePool extends Operator {

  def GlobalAveragePool1(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait Softsign extends Operator {

  def Softsign1(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait DepthToSpace extends Operator {

  def DepthToSpace1(name: String,input: example.T, inputname: String,blocksize : (String))
    : FS[(example.T)]

}
@free trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp1(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Cos extends Operator {

  def Cos7(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait Clip extends Operator {

  def Clip1(name: String,input: example.T, inputname: String,consumed_inputs : Option[(Seq[String])] = None,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(example.T)]


  def Clip6(name: String,input: example.T, inputname: String,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Crop extends Operator {

  def Crop1(name: String,input: example.T, inputname: String,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait SpaceToDepth extends Operator {

  def SpaceToDepth1(name: String,input: example.T, inputname: String,blocksize : (String))
    : FS[(example.T)]

}
@free trait Exp extends Operator {

  def Exp1(name: String,input: example.T, inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Exp6(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait Greater extends Operator {

  def Greater1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T1)]


  def Greater7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T1)]

}
@free trait ReduceMin extends Operator {

  def ReduceMin1(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ParametricSoftplus extends Operator {

  def ParametricSoftplus1(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Affine extends Operator {

  def Affine1(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait ImageScaler extends Operator {

  def ImageScaler1(name: String,input: example.T, inputname: String,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Relu extends Operator {

  def Relu1(name: String,X: example.T, Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Relu6(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait Atan extends Operator {

  def Atan7(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}
@free trait Sub extends Operator {

  def Sub1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Sub6(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T)]


  def Sub7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T)]

}
@free trait Pow extends Operator {

  def Pow1(name: String,X: example.T, Xname: String, Y: example.T, Yname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T)]


  def Pow7(name: String,X: example.T, Xname: String, Y: example.T, Yname: String)
    : FS[(example.T)]

}
@free trait Conv extends Operator {

  def Conv1(name: String,X: example.T, Xname: String, W: example.T, Wname: String,B: Option[example.T] = None,auto_pad : Option[(example.Tensor[Number])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait Cast extends Operator {

  def Cast1(name: String,input: example.T1, inputname: String,to : (example.Tensor[Number]))
    : FS[(example.T2)]


  def Cast6(name: String,input: example.T1, inputname: String,to : (String))
    : FS[(example.T2)]

}
@free trait RandomNormal extends Operator {

  def RandomNormal1(name: String)
    : FS[(example.T)]

}
@free trait ConvTranspose extends Operator {

  def ConvTranspose1(name: String,X: example.T, Xname: String, W: example.T, Wname: String,B: Option[example.T] = None,auto_pad : Option[(example.Tensor[Number])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait ConstantFill extends Operator {

  def ConstantFill1(name: String,input: Option[example.T1] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : FS[(example.T2)]

}
@free trait Unsqueeze extends Operator {

  def Unsqueeze1(name: String,data: example.T, dataname: String,axes : (Seq[String]))
    : FS[(example.T)]

}
@free trait Neg extends Operator {

  def Neg1(name: String,X: example.T, Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Neg6(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait ArgMax extends Operator {

  def ArgMax1(name: String,data: example.T, dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(example.Tensor[Long])]

}
@free trait If extends Operator {

  def If1(name: String,cond: example.B, condname: String,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : FS[(example.V)]

}
@free trait Selu extends Operator {

  def Selu1(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None,gamma : Option[(Int)] = None)
    : FS[(example.T)]


  def Selu6(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : FS[(example.T)]

}
@free trait Split extends Operator {

  def Split1(name: String,input: example.T, inputname: String,split: Option[example.T] = None,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Split2(name: String,input: example.T, inputname: String,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait GRU extends Operator {

  def GRU1(name: String,X: example.T, Xname: String, W: example.T, Wname: String, R: example.T, Rname: String,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(example.T, example.T)]


  def GRU3(name: String,X: example.T, Xname: String, W: example.T, Wname: String, R: example.T, Rname: String,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(example.T, example.T)]


  def GRU7(name: String,X: example.T, Xname: String, W: example.T, Wname: String, R: example.T, Rname: String,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : FS[(example.T, example.T)]

}
@free trait BatchNormalization extends Operator {

  def BatchNormalization1(name: String,X: example.T, Xname: String, scale: example.T, scalename: String, B: example.T, Bname: String, mean: example.T, meanname: String, someVar: example.T, varname: String,consumed_inputs : (Seq[String]),epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(example.T, example.T, example.T, example.T, example.T)]


  def BatchNormalization6(name: String,X: example.T, Xname: String, scale: example.T, scalename: String, B: example.T, Bname: String, mean: example.T, meanname: String, someVar: example.T, varname: String,epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(example.T, example.T, example.T, example.T, example.T)]


  def BatchNormalization7(name: String,X: example.T, Xname: String, scale: example.T, scalename: String, B: example.T, Bname: String, mean: example.T, meanname: String, someVar: example.T, varname: String,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(example.T, example.T, example.T, example.T, example.T)]

}
@free trait Upsample extends Operator {

  def Upsample1(name: String,X: example.T, Xname: String,height_scaleAttr : (Int),mode : Option[(example.Tensor[Number])] = None,width_scaleAttr : (Int))
    : FS[(example.T)]


  def Upsample7(name: String,X: example.T, Xname: String,mode : Option[(example.Tensor[Number])] = None,scaleAttrs : (Seq[Int]))
    : FS[(example.T)]

}
@free trait Mul extends Operator {

  def Mul1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Mul6(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T)]


  def Mul7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T)]

}
@free trait Sqrt extends Operator {

  def Sqrt1(name: String,X: example.T, Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Sqrt6(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait Add extends Operator {

  def Add1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Add6(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T)]


  def Add7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T)]

}
@free trait Sigmoid extends Operator {

  def Sigmoid1(name: String,X: example.T, Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Sigmoid6(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait GlobalLpPool extends Operator {

  def GlobalLpPool1(name: String,X: example.T, Xname: String,p : Option[(Int)] = None)
    : FS[(example.T)]


  def GlobalLpPool2(name: String,X: example.T, Xname: String,p : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Or extends Operator {

  def Or1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T1)]


  def Or7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T1)]

}
@free trait LRN extends Operator {

  def LRN1(name: String,X: example.T, Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : FS[(example.T)]

}
@free trait Xor extends Operator {

  def Xor1(name: String,A: example.T, Aname: String, B: example.T, Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(example.T1)]


  def Xor7(name: String,A: example.T, Aname: String, B: example.T, Bname: String)
    : FS[(example.T1)]

}
@free trait Max extends Operator {

  def Max1(name: String)
    : FS[(example.T)]


  def Max6(name: String)
    : FS[(example.T)]

}
@free trait Slice extends Operator {

  def Slice1(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : FS[(example.T)]

}
@free trait LSTM extends Operator {

  def LSTM1(name: String,X: example.T, Xname: String, W: example.T, Wname: String, R: example.T, Rname: String,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None, initial_c: Option[example.T] = None, P: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(example.T, example.T, example.T)]


  def LSTM7(name: String,X: example.T, Xname: String, W: example.T, Wname: String, R: example.T, Rname: String,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None, initial_c: Option[example.T] = None, P: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : FS[(example.T, example.T, example.T)]

}
@free trait LoopIndexTensor extends Operator {

  def LoopIndexTensor1(name: String,T: example.T, Tname: String, loop_idx: example.I, loop_idxname: String,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Floor extends Operator {

  def Floor1(name: String,X: example.T, Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Floor6(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait RandomUniform extends Operator {

  def RandomUniform1(name: String)
    : FS[(example.T)]

}
@free trait MaxPool extends Operator {

  def MaxPool1(name: String,X: example.T, Xname: String,auto_pad : Option[(example.Tensor[Number])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(example.T)]

}
@free trait Ceil extends Operator {

  def Ceil1(name: String,X: example.T, Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Ceil6(name: String,X: example.T, Xname: String)
    : FS[(example.T)]

}
@free trait ReduceL2 extends Operator {

  def ReduceL21(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait ReduceLogSum extends Operator {

  def ReduceLogSum1(name: String,data: example.T, dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait RNN extends Operator {

  def RNN1(name: String,X: example.T, Xname: String, W: example.T, Wname: String, R: example.T, Rname: String,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(example.T, example.T)]


  def RNN7(name: String,X: example.T, Xname: String, W: example.T, Wname: String, R: example.T, Rname: String,B: Option[example.T] = None, sequence_lens: Option[example.T1] = None, initial_h: Option[example.T] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[example.Tensor[Number]])] = None,clip : Option[(Int)] = None,direction : Option[(example.Tensor[Number])] = None,hidden_size : Option[(String)] = None)
    : FS[(example.T, example.T)]

}
@free trait Reshape extends Operator {

  def Reshape1(name: String,data: example.T, dataname: String,consumed_inputs : Option[(Seq[String])] = None,shape : Option[(Seq[String])] = None)
    : FS[(example.T)]


  def Reshape5(name: String,data: example.T, dataname: String, shape: example.Tensor[Long], shapename: String)
    : FS[(example.T)]

}
@free trait Mean extends Operator {

  def Mean1(name: String)
    : FS[(example.T)]


  def Mean6(name: String)
    : FS[(example.T)]

}
@free trait LogSoftmax extends Operator {

  def LogSoftmax1(name: String,input: example.T, inputname: String,axis : Option[(String)] = None)
    : FS[(example.T)]

}
@free trait Loop extends Operator {

  def Loop1(name: String,M: example.I, Mname: String, cond: example.B, condname: String,body : (Seq[Float]))
    : FS[(example.V)]

}
@free trait Asin extends Operator {

  def Asin7(name: String,input: example.T, inputname: String)
    : FS[(example.T)]

}