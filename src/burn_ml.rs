impl<E:Element,K:Basic,const N:usize> TryFrom<BuiltinTensor<E>> for Tensor<N,K>{
	fn try_from(tensor:BuiltinTensor<E>)->Result<Self,Self::Error>{tensor.view().try_into()}
	type Error=TensorError;
}
impl<E:Element,K:Basic,const N:usize> TryFrom<Tens<E>> for Tensor<N,K>{
	fn try_from(tensor:Tens<E>)->Result<Self,Self::Error>{tensor.view().try_into()}
	type Error=TensorError;
}
impl<E:Element,K:Basic,const N:usize> TryFrom<Tensor<N,K>> for BuiltinTensor<E>{
	fn try_from(value:Tensor<N,K>)->Result<Self,Self::Error>{
		let data=value.to_data();
		let layout=Layout::new(&data.shape());

		let data=data.try_to_vec()?;
		Ok(Tens::from_inner(data,layout).tensor())
	}
	type Error=DataError;
}
impl<E:Element,K:Basic,const N:usize> TryFrom<Tensor<N,K>> for Tens<E>{
	fn try_from(value:Tensor<N,K>)->Result<Self,Self::Error>{Ok(BuiltinTensor::try_from(value)?.tens())}
	type Error=DataError;
}
impl<E:Element,K:Basic,const N:usize> TryFrom<&View<E>> for Tensor<N,K>{
	fn try_from(tensor:&View<E>)->Result<Self,Self::Error>{
		tensor.validate()?;

		if tensor.rank()!=N{return Err(TensorError::specific_rank_mismatch(tensor.get_layout(),N,"convert"))}
		let dims=tensor.dims().to_vec();

		let data=tensor.flat_vec(None);
		let device=&Default::default();
		let tensordata=TensorData::new(data,dims);

		Ok(Tensor::from_data(tensordata, device))
	}
	type Error=TensorError;
}

#[cfg(feature="serial")]
/// deserialize a burn bool tensor from intertense tensor format with a specific component type bool not necessarily given by the kind
pub fn deserialize_bool_tensor<'a,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<N,Bool>,D::Error>{deserialize_elem_tensor::<D,bool,Bool,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn float tensor from intertense tensor format with a specific component type f64 not necessarily given by the kind
pub fn deserialize_double_tensor<'a,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<N,Float>,D::Error>{deserialize_elem_tensor::<D,f64,Float,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn tensor from intertense tensor format with a specific component type not necessarily given by the kind
pub fn deserialize_elem_tensor<'a,D:Deserializer<'a>,E:Deserialize<'a>+Element,K:Basic,const N:usize>(deserializer:D)->Result<Tensor<N,K>,D::Error>{
	let tensor:Tens<E>=Tens::deserialize(deserializer)?;
	if tensor.rank()!=N{return Err(Derror::custom(format!("rank mismatch")))}

	let dims=tensor.dims().to_vec();

	let data=tensor.into_flat_vec();
	let device=&Default::default();
	let tensordata=TensorData::new(data,dims);

	Ok(Tensor::from_data(tensordata, device))
}
#[cfg(feature="serial")]
/// deserialize a burn float tensor from intertense tensor format with a specific component type f32 not necessarily given by the kind
pub fn deserialize_float_tensor<'a,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<N,Float>,D::Error>{deserialize_elem_tensor::<D,f32,Float,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn int tensor from intertense tensor format with a specific component type i32 not necessarily given by the kind
pub fn deserialize_int_tensor<'a,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<N,Int>,D::Error>{deserialize_elem_tensor::<D,i32,Int,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn int tensor from intertense tensor format with a specific component type i64 not necessarily given by the kind
pub fn deserialize_long_tensor<'a,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<N,Int>,D::Error>{deserialize_elem_tensor::<D,i64,Int,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn int tensor from intertense tensor format with a specific component type u32 not necessarily given by the kind
pub fn deserialize_uint_tensor<'a,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<N,Int>,D::Error>{deserialize_elem_tensor::<D,u32,Int,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn int tensor from intertense tensor format with a specific component type u64 not necessarily given by the kind
pub fn deserialize_ulong_tensor<'a,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<N,Int>,D::Error>{deserialize_elem_tensor::<D,u64,Int,N>(deserializer)}
#[cfg(feature="serial")]
/// serialize a burn bool tensor to intertense tensor format with a specific component type bool not necessarily given by the kind
pub fn serialize_bool_tensor<S:Serializer,const N:usize>(tensor:&Tensor<N,Bool>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<bool,Bool,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn float tensor to intertense tensor format with a specific component type f64 not necessarily given by the kind
pub fn serialize_double_tensor<S:Serializer,const N:usize>(tensor:&Tensor<N,Float>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<f64,Float,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn tensor to intertense tensor format with a specific component type not necessarily given by the kind
pub fn serialize_elem_tensor<E:Element+Serialize,K:Basic,S:Serializer,const N:usize>(tensor:&Tensor<N,K>,serializer:S)->Result<S::Ok,S::Error>{
	let data=tensor.to_data();
	let layout=Layout::new(&data.shape());

	let data:Vec<E>=data.convert::<E>().try_to_vec().map_err(|e|Serror::custom(format!("{e:?}")))?;
	Tens::from_inner(data,layout).serialize(serializer)
}
#[cfg(feature="serial")]
/// serialize a burn float tensor to intertense tensor format with a specific component type f32 not necessarily given by the kind
pub fn serialize_float_tensor<S:Serializer,const N:usize>(tensor:&Tensor<N,Float>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<f32,Float,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn int tensor to intertense tensor format with a specific component type i32 not necessarily given by the kind
pub fn serialize_int_tensor<S:Serializer,const N:usize>(tensor:&Tensor<N,Int>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<i32,Int,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn int tensor to intertense tensor format with a specific component type i64 not necessarily given by the kind
pub fn serialize_long_tensor<S:Serializer,const N:usize>(tensor:&Tensor<N,Int>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<i64,Int,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn int tensor to intertense tensor format with a specific component type u32 not necessarily given by the kind
pub fn serialize_uint_tensor<S:Serializer,const N:usize>(tensor:&Tensor<N,Int>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<u32,Int,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn int tensor to intertense tensor format with a specific component type u64 not necessarily given by the kind
pub fn serialize_ulong_tensor<S:Serializer,const N:usize>(tensor:&Tensor<N,Int>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<u64,Int,S,N>(tensor,serializer)}

use burn::{
	prelude::*,tensor::{DataError,Element,kind::Basic}
};
use crate::builtin_tensor::{Error as TensorError,Layout,Tens,tensor::Tensor as BuiltinTensor,view::View};
#[cfg(feature="serial")]
use serde::{Deserialize,Deserializer,Serialize,Serializer,de::Error as Derror,ser::Error as Serror};

