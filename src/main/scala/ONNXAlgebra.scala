package org.emergentorder

import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Numeric
import scala.reflect.ClassTag

package object onnx {
  type Tensor[U] = Tuple2[Vector[U], Seq[Int]]
  trait Operator
  type TTensorFloat16 = Tensor[Float16]

  type T1TensorInt64 = Tensor[Long]

  type T1Mapstringdouble =Map[String, Double]

  type VTensorInt64 = Tensor[Long]

  type TSeqmapstringfloat =Seq[Map[String, Float]]

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

  type TSeqmapint64float =Seq[Map[Long, Float]]

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

  type T1Mapint64string =Map[Long, String]

  type T2TensorString = Tensor[String]

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

  type T1Mapint64double =Map[Long, Double]

  type VTensorUint16 = Tensor[UShort]

  type T1TensorUint32 = Tensor[UInt]

  type T1Mapstringint64 =Map[String, Long]

  type VTensorFloat = Tensor[Float]

  type T1TensorBool = Tensor[Boolean]

  type T1Mapint64float =Map[Long, Float]

  type VTensorUint8 = Tensor[UByte]

  type T1Mapstringfloat =Map[String, Float]

  type VTensorFloat16 = Tensor[Float16]

  type TTensorInt64 = Tensor[Long]

  type TindTensorInt32 = Tensor[Int]
  type T[VV] = Tensor[VV]
  type t1[VV] = Tensor[VV]
  type V[VV] = Tensor[VV]
  type t2[VV] = Tensor[VV]
  type B[VV] = Tensor[VV]
  type tind[VV] = Tensor[VV]
  type I[VV] = Tensor[VV]
trait DataSource {
  def inputData[VV:Numeric:ClassTag]: Tensor[VV]
  def getParams[VV:Numeric:ClassTag](name: String): Tensor[VV]
  def getAttributes[VV:Numeric:ClassTag](name: String): Tensor[VV]
}
trait InstanceNormalization extends Operator {

  def InstanceNormalization1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String,consumed_inputs : Option[(Seq[String])] = None,epsilon : Option[(Int)] = None)
    : (T[VV])


  def InstanceNormalization6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String,epsilon : Option[(Int)] = None)
    : (T[VV])

}
trait LpPool extends Operator {

  def LpPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,kernel_shape : Option[(Seq[String])] = None,p : Option[(Int)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (T[VV])


  def LpPool2[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (T[VV])

}
trait ReduceMean extends Operator {

  def ReduceMean1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait LinearClassifier extends Operator {

  def LinearClassifier1[VV : Numeric:ClassTag](name: String,X: t1[VV], Xname: String,classlabels_ints : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[VV]])] = None,coefficients : (Seq[Int]),intercepts : Option[(Seq[Int])] = None,multi_class : Option[(String)] = None,post_transform : Option[(Tensor[VV])] = None)
    : (t2[VV], Tensor[Float])

}
trait LabelEncoder extends Operator {

  def LabelEncoder1[VV : Numeric:ClassTag](name: String,X: t1[VV], Xname: String,classes_strings : Option[(Seq[Tensor[VV]])] = None,default_int64 : Option[(String)] = None,default_string : Option[(Tensor[VV])] = None)
    : (t2[VV])

}
trait Identity extends Operator {

  def Identity1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait MaxRoiPool extends Operator {

  def MaxRoiPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, rois: T[VV], roisname: String,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : (T[VV])

}
trait Affine extends Operator {

  def Affine1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (T[VV])

}
trait Elu extends Operator {

  def Elu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Elu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None)
    : (T[VV])

}
trait Equal extends Operator {

  def Equal1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (t1[VV])


  def Equal7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (t1[VV])

}
trait RandomUniform extends Operator {

  def RandomUniform1[VV : Numeric:ClassTag](name: String)
    : (T[VV])

}
trait Softplus extends Operator {

  def Softplus1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait Gather extends Operator {

  def Gather1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String, indices: tind[VV], indicesname: String,axis : Option[(String)] = None)
    : (T[VV])

}
trait Pow extends Operator {

  def Pow1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, Y: T[VV], Yname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (T[VV])


  def Pow7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, Y: T[VV], Yname: String)
    : (T[VV])

}
trait Tanh extends Operator {

  def Tanh1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Tanh6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait Selu extends Operator {

  def Selu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None,gamma : Option[(Int)] = None)
    : (T[VV])


  def Selu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : (T[VV])

}
trait ConvTranspose extends Operator {

  def ConvTranspose1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String,B: Option[T[VV]] = None,auto_pad : Option[(Tensor[VV])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (T[VV])

}
trait Max extends Operator {

  def Max1[VV : Numeric:ClassTag](name: String)
    : (T[VV])


  def Max6[VV : Numeric:ClassTag](name: String)
    : (T[VV])

}
trait Neg extends Operator {

  def Neg1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Neg6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait Shape extends Operator {

  def Shape1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String)
    : (t1[VV])

}
trait TreeEnsembleClassifier extends Operator {

  def TreeEnsembleClassifier1[VV : Numeric:ClassTag](name: String,X: t1[VV], Xname: String,base_values : Option[(Seq[Int])] = None,class_ids : Option[(Seq[String])] = None,class_nodeids : Option[(Seq[String])] = None,class_treeids : Option[(Seq[String])] = None,class_weights : Option[(Seq[Int])] = None,classlabels_int64s : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[VV]])] = None,nodes_falsenodeids : Option[(Seq[String])] = None,nodes_featureids : Option[(Seq[String])] = None,nodes_hitrates : Option[(Seq[Int])] = None,nodes_missing_value_tracks_true : Option[(Seq[String])] = None,nodes_modes : Option[(Seq[Tensor[VV]])] = None,nodes_nodeids : Option[(Seq[String])] = None,nodes_treeids : Option[(Seq[String])] = None,nodes_truenodeids : Option[(Seq[String])] = None,nodes_values : Option[(Seq[Int])] = None,post_transform : Option[(Tensor[VV])] = None)
    : (t2[VV], Tensor[Float])

}
trait DepthToSpace extends Operator {

  def DepthToSpace1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,blocksize : (String))
    : (T[VV])

}
trait Loop extends Operator {

  def Loop1[VV : Numeric:ClassTag](name: String,M: I[VV], Mname: String, cond: B[VV], condname: String,body : (Seq[Float]))
    : (V[VV])

}
trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : (T[VV])

}
trait Add extends Operator {

  def Add1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Add6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (T[VV])


  def Add7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (T[VV])

}
trait LSTM extends Operator {

  def LSTM1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[t1[VV]] = None, initial_h: Option[T[VV]] = None, initial_c: Option[T[VV]] = None, P: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (T[VV], T[VV], T[VV])


  def LSTM7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[t1[VV]] = None, initial_h: Option[T[VV]] = None, initial_c: Option[T[VV]] = None, P: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : (T[VV], T[VV], T[VV])

}
trait Dropout extends Operator {

  def Dropout1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,consumed_inputs : Option[(Seq[String])] = None,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : (T[VV], T[VV])


  def Dropout6[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : (T[VV], T[VV])


  def Dropout7[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,ratio : Option[(Int)] = None)
    : (T[VV], T[VV])

}
trait Sqrt extends Operator {

  def Sqrt1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Sqrt6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait RandomUniformLike extends Operator {

  def RandomUniformLike1[VV : Numeric:ClassTag](name: String,input: t1[VV], inputname: String,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : (t2[VV])

}
trait Transpose extends Operator {

  def Transpose1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,perm : Option[(Seq[String])] = None)
    : (T[VV])

}
trait ReduceMax extends Operator {

  def ReduceMax1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait GlobalMaxPool extends Operator {

  def GlobalMaxPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait Less extends Operator {

  def Less1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (t1[VV])


  def Less7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (t1[VV])

}
trait Upsample extends Operator {

  def Upsample1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,height_scaleAttr : (Int),mode : Option[(Tensor[VV])] = None,width_scaleAttr : (Int))
    : (T[VV])


  def Upsample7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,mode : Option[(Tensor[VV])] = None,scaleAttrs : (Seq[Int]))
    : (T[VV])

}
trait Min extends Operator {

  def Min1[VV : Numeric:ClassTag](name: String)
    : (T[VV])


  def Min6[VV : Numeric:ClassTag](name: String)
    : (T[VV])

}
trait GivenTensorFill extends Operator {

  def GivenTensorFill1[VV : Numeric:ClassTag](name: String,shapeInput: Option[T[VV]] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : (T[VV])

}
trait Squeeze extends Operator {

  def Squeeze1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : (Seq[String]))
    : (T[VV])

}
trait LinearRegressor extends Operator {

  def LinearRegressor1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,coefficients : Option[(Seq[Int])] = None,intercepts : Option[(Seq[Int])] = None,post_transform : Option[(Tensor[VV])] = None,targets : Option[(String)] = None)
    : (Tensor[Float])

}
trait Tan extends Operator {

  def Tan7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait GlobalLpPool extends Operator {

  def GlobalLpPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,p : Option[(Int)] = None)
    : (T[VV])


  def GlobalLpPool2[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,p : Option[(String)] = None)
    : (T[VV])

}
trait Reshape extends Operator {

  def Reshape1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,consumed_inputs : Option[(Seq[String])] = None,shape : Option[(Seq[String])] = None)
    : (T[VV])


  def Reshape5[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String, shape: Tensor[Long], shapename: String)
    : (T[VV])

}
trait RandomNormalLike extends Operator {

  def RandomNormalLike1[VV : Numeric:ClassTag](name: String,input: t1[VV], inputname: String,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : (t2[VV])

}
trait ArgMin extends Operator {

  def ArgMin1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : (Tensor[Long])

}
trait LeakyRelu extends Operator {

  def LeakyRelu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def LeakyRelu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None)
    : (T[VV])

}
trait Softmax extends Operator {

  def Softmax1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None)
    : (T[VV])

}
trait GRU extends Operator {

  def GRU1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[t1[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (T[VV], T[VV])


  def GRU3[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[t1[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (T[VV], T[VV])


  def GRU7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[t1[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : (T[VV], T[VV])

}
trait LRN extends Operator {

  def LRN1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : (T[VV])

}
trait ReduceL1 extends Operator {

  def ReduceL11[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait Binarizer extends Operator {

  def Binarizer1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,threshold : Option[(Int)] = None)
    : (T[VV])

}
trait GRUUnit extends Operator {

  def GRUUnit1[VV : Numeric:ClassTag](name: String,hidden_prev: T[VV], hidden_prevname: String, gates: T[VV], gatesname: String, seq_lengths: T[VV], seq_lengthsname: String, t: T[VV], tname: String,drop_states : Option[(String)] = None)
    : (T[VV])

}
trait Clip extends Operator {

  def Clip1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,consumed_inputs : Option[(Seq[String])] = None,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : (T[VV])


  def Clip6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : (T[VV])

}
trait FeatureVectorizer extends Operator {

  def FeatureVectorizer1[VV : Numeric:ClassTag](name: String)
    : (Tensor[Float])

}
trait Imputer extends Operator {

  def Imputer1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,imputed_value_floats : Option[(Seq[Int])] = None,imputed_value_int64s : Option[(Seq[String])] = None,replaced_value_float : Option[(Int)] = None,replaced_value_int64 : Option[(String)] = None)
    : (T[VV])

}
trait MaxPool extends Operator {

  def MaxPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (T[VV])

}
trait Softsign extends Operator {

  def Softsign1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait ZipMap extends Operator {

  def ZipMap1[VV : Numeric:ClassTag](name: String,X: Tensor[Float], Xname: String,classlabels_int64s : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[VV]])] = None)
    : (T[VV])

}
trait HardSigmoid extends Operator {

  def HardSigmoid1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def HardSigmoid6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (T[VV])

}
trait ReduceSumSquare extends Operator {

  def ReduceSumSquare1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait SVMRegressor extends Operator {

  def SVMRegressor1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,coefficients : Option[(Seq[Int])] = None,kernel_params : Option[(Seq[Int])] = None,kernel_type : Option[(Tensor[VV])] = None,n_supports : Option[(String)] = None,one_class : Option[(String)] = None,post_transform : Option[(Tensor[VV])] = None,rho : Option[(Seq[Int])] = None,support_vectors : Option[(Seq[Int])] = None)
    : (Tensor[Float])

}
trait ScaledTanh extends Operator {

  def ScaledTanh1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (T[VV])

}
trait Normalizer extends Operator {

  def Normalizer1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,norm : Option[(Tensor[VV])] = None)
    : (Tensor[Float])

}
trait Pad extends Operator {

  def Pad1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,mode : Option[(Tensor[VV])] = None,paddings : (Seq[String]),value : Option[(Int)] = None)
    : (T[VV])


  def Pad2[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,mode : Option[(Tensor[VV])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : (T[VV])

}
trait Unsqueeze extends Operator {

  def Unsqueeze1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : (Seq[String]))
    : (T[VV])

}
trait ReduceSum extends Operator {

  def ReduceSum1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait OneHotEncoder extends Operator {

  def OneHotEncoder1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,cats_int64s : Option[(Seq[String])] = None,cats_strings : Option[(Seq[Tensor[VV]])] = None,zeros : Option[(String)] = None)
    : (Tensor[Float])

}
trait ParametricSoftplus extends Operator {

  def ParametricSoftplus1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (T[VV])

}
trait SVMClassifier extends Operator {

  def SVMClassifier1[VV : Numeric:ClassTag](name: String,X: t1[VV], Xname: String,classlabels_ints : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[VV]])] = None,coefficients : Option[(Seq[Int])] = None,kernel_params : Option[(Seq[Int])] = None,kernel_type : Option[(Tensor[VV])] = None,post_transform : Option[(Tensor[VV])] = None,prob_a : Option[(Seq[Int])] = None,prob_b : Option[(Seq[Int])] = None,rho : Option[(Seq[Int])] = None,support_vectors : Option[(Seq[Int])] = None,vectors_per_class : Option[(Seq[String])] = None)
    : (t2[VV], Tensor[Float])

}
trait Tile extends Operator {

  def Tile1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String, tiles: T[VV], tilesname: String, axis: T[VV], axisname: String)
    : (T[VV])


  def Tile6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String, repeats: t1[VV], repeatsname: String)
    : (T[VV])

}
trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait CategoryMapper extends Operator {

  def CategoryMapper1[VV : Numeric:ClassTag](name: String,X: t1[VV], Xname: String,cats_int64s : Option[(Seq[String])] = None,cats_strings : Option[(Seq[Tensor[VV]])] = None,default_int64 : Option[(String)] = None,default_string : Option[(Tensor[VV])] = None)
    : (t2[VV])

}
trait Exp extends Operator {

  def Exp1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Exp6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait Reciprocal extends Operator {

  def Reciprocal1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Reciprocal6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait Gemm extends Operator {

  def Gemm1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String, C: T[VV], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : (T[VV])


  def Gemm6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String, C: T[VV], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : (T[VV])


  def Gemm7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String, C: T[VV], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : (T[VV])

}
trait TopK extends Operator {

  def TopK1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,axis : Option[(String)] = None,k : (String))
    : (T[VV], I[VV])

}
trait ReduceL2 extends Operator {

  def ReduceL21[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait Xor extends Operator {

  def Xor1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (t1[VV])


  def Xor7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (t1[VV])

}
trait Cast extends Operator {

  def Cast1[VV : Numeric:ClassTag](name: String,input: t1[VV], inputname: String,to : (Tensor[VV]))
    : (t2[VV])


  def Cast6[VV : Numeric:ClassTag](name: String,input: t1[VV], inputname: String,to : (String))
    : (t2[VV])

}
trait ArrayFeatureExtractor extends Operator {

  def ArrayFeatureExtractor1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, Y: Tensor[Long], Yname: String)
    : (T[VV])

}
trait LpNormalization extends Operator {

  def LpNormalization1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None,p : Option[(String)] = None)
    : (T[VV])

}
trait Abs extends Operator {

  def Abs1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Abs6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait Log extends Operator {

  def Log1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Log6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait Ceil extends Operator {

  def Ceil1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Ceil6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait LoopIndexTensor extends Operator {

  def LoopIndexTensor1[VV : Numeric:ClassTag](name: String,T: T[VV], Tname: String, loop_idx: I[VV], loop_idxname: String,axis : Option[(String)] = None)
    : (T[VV])

}
trait Sub extends Operator {

  def Sub1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Sub6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (T[VV])


  def Sub7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (T[VV])

}
trait Size extends Operator {

  def Size1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String)
    : (t1[VV])

}
trait Relu extends Operator {

  def Relu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Relu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait Div extends Operator {

  def Div1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Div6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (T[VV])


  def Div7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (T[VV])

}
trait DictVectorizer extends Operator {

  def DictVectorizer1[VV : Numeric:ClassTag](name: String,X: t1[VV], Xname: String,int64_vocabulary : Option[(Seq[String])] = None,string_vocabulary : Option[(Seq[Tensor[VV]])] = None)
    : (t2[VV])

}
trait Scaler extends Operator {

  def Scaler1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,offset : Option[(Seq[Int])] = None,scaleAttr : Option[(Seq[Int])] = None)
    : (Tensor[Float])

}
trait ThresholdedRelu extends Operator {

  def ThresholdedRelu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None)
    : (T[VV])

}
trait LogSoftmax extends Operator {

  def LogSoftmax1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None)
    : (T[VV])

}
trait BatchNormalization extends Operator {

  def BatchNormalization1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String, mean: T[VV], meanname: String, someVar: T[VV], varname: String,consumed_inputs : (Seq[String]),epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : (T[VV], T[VV], T[VV], T[VV], T[VV])


  def BatchNormalization6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String, mean: T[VV], meanname: String, someVar: T[VV], varname: String,epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : (T[VV], T[VV], T[VV], T[VV], T[VV])


  def BatchNormalization7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String, mean: T[VV], meanname: String, someVar: T[VV], varname: String,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : (T[VV], T[VV], T[VV], T[VV], T[VV])

}
trait CastMap extends Operator {

  def CastMap1[VV : Numeric:ClassTag](name: String,X: t1[VV], Xname: String,cast_to : Option[(Tensor[VV])] = None,map_form : Option[(Tensor[VV])] = None,max_map : Option[(String)] = None)
    : (t2[VV])

}
trait AveragePool extends Operator {

  def AveragePool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (T[VV])


  def AveragePool7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (T[VV])

}
trait ReduceProd extends Operator {

  def ReduceProd1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait Crop extends Operator {

  def Crop1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : (T[VV])

}
trait Mean extends Operator {

  def Mean1[VV : Numeric:ClassTag](name: String)
    : (T[VV])


  def Mean6[VV : Numeric:ClassTag](name: String)
    : (T[VV])

}
trait And extends Operator {

  def And1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (t1[VV])


  def And7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (t1[VV])

}
trait Split extends Operator {

  def Split1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,split: Option[T[VV]] = None,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : (T[VV])


  def Split2[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : (T[VV])

}
trait ArgMax extends Operator {

  def ArgMax1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : (Tensor[Long])

}
trait ConstantFill extends Operator {

  def ConstantFill1[VV : Numeric:ClassTag](name: String,input: Option[t1[VV]] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : (t2[VV])

}
trait GlobalAveragePool extends Operator {

  def GlobalAveragePool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait Sum extends Operator {

  def Sum1[VV : Numeric:ClassTag](name: String)
    : (T[VV])


  def Sum6[VV : Numeric:ClassTag](name: String)
    : (T[VV])

}
trait Greater extends Operator {

  def Greater1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (t1[VV])


  def Greater7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (t1[VV])

}
trait Or extends Operator {

  def Or1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (t1[VV])


  def Or7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (t1[VV])

}
trait TreeEnsembleRegressor extends Operator {

  def TreeEnsembleRegressor1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,aggregate_function : Option[(Tensor[VV])] = None,base_values : Option[(Seq[Int])] = None,n_targets : Option[(String)] = None,nodes_falsenodeids : Option[(Seq[String])] = None,nodes_featureids : Option[(Seq[String])] = None,nodes_hitrates : Option[(Seq[Int])] = None,nodes_missing_value_tracks_true : Option[(Seq[String])] = None,nodes_modes : Option[(Seq[Tensor[VV]])] = None,nodes_nodeids : Option[(Seq[String])] = None,nodes_treeids : Option[(Seq[String])] = None,nodes_truenodeids : Option[(Seq[String])] = None,nodes_values : Option[(Seq[Int])] = None,post_transform : Option[(Tensor[VV])] = None,target_ids : Option[(Seq[String])] = None,target_nodeids : Option[(Seq[String])] = None,target_treeids : Option[(Seq[String])] = None,target_weights : Option[(Seq[Int])] = None)
    : (Tensor[Float])

}
trait Constant extends Operator {

  def Constant1[VV : Numeric:ClassTag](name: String)
    : (T[VV])

}
trait Slice extends Operator {

  def Slice1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : (T[VV])

}
trait Asin extends Operator {

  def Asin7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait MatMul extends Operator {

  def MatMul1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (T[VV])

}
trait Not extends Operator {

  def Not1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait RNN extends Operator {

  def RNN1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[t1[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (T[VV], T[VV])


  def RNN7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[t1[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None)
    : (T[VV], T[VV])

}
trait Sin extends Operator {

  def Sin7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait ImageScaler extends Operator {

  def ImageScaler1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : (T[VV])

}
trait Mul extends Operator {

  def Mul1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Mul6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (T[VV])


  def Mul7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : (T[VV])

}
trait Scale extends Operator {

  def Scale1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,scaleAttr : Option[(Int)] = None)
    : (T[VV])

}
trait ReduceLogSum extends Operator {

  def ReduceLogSum1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait Atan extends Operator {

  def Atan7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait Concat extends Operator {

  def Concat1[VV : Numeric:ClassTag](name: String)
    : (T[VV])


  def Concat4[VV : Numeric:ClassTag](name: String)
    : (T[VV])

}
trait ReduceMin extends Operator {

  def ReduceMin1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (T[VV])

}
trait Conv extends Operator {

  def Conv1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String,B: Option[T[VV]] = None,auto_pad : Option[(Tensor[VV])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (T[VV])

}
trait SpaceToDepth extends Operator {

  def SpaceToDepth1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,blocksize : (String))
    : (T[VV])

}
trait PRelu extends Operator {

  def PRelu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, slope: T[VV], slopename: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def PRelu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, slope: T[VV], slopename: String)
    : (T[VV])


  def PRelu7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, slope: T[VV], slopename: String)
    : (T[VV])

}
trait Multinomial extends Operator {

  def Multinomial7[VV : Numeric:ClassTag](name: String,input: t1[VV], inputname: String,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : (t2[VV])

}
trait Flatten extends Operator {

  def Flatten1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None)
    : (T[VV])

}
trait Floor extends Operator {

  def Floor1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Floor6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait Hardmax extends Operator {

  def Hardmax1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None)
    : (T[VV])

}
trait Cos extends Operator {

  def Cos7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}
trait If extends Operator {

  def If1[VV : Numeric:ClassTag](name: String,cond: B[VV], condname: String,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : (V[VV])

}
trait Sigmoid extends Operator {

  def Sigmoid1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (T[VV])


  def Sigmoid6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : (T[VV])

}
trait RandomNormal extends Operator {

  def RandomNormal1[VV : Numeric:ClassTag](name: String)
    : (T[VV])

}
trait Acos extends Operator {

  def Acos7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : (T[VV])

}}
