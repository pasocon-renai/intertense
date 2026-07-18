impl<B:Backend,K:BasicOps<B>,const N:usize> TryFrom<BuiltinTensor<K::Elem>> for Tensor<B,N,K>{
	fn try_from(tensor:BuiltinTensor<K::Elem>)->Result<Self,Self::Error>{
		if tensor.rank()!=N{return Err(TensorError::specific_rank_mismatch(tensor.get_layout(),N,"convert"))}
		let dims=tensor.dims().to_vec();

		let data=tensor.flat_vec(None);
		let device=&Default::default();
		let tensordata=TensorData::new(data,dims);

		Ok(Tensor::from_data(tensordata, device))
	}
	type Error=TensorError;
}
impl<B:Backend,K:BasicOps<B>,const N:usize> TryFrom<Tens<K::Elem>> for Tensor<B,N,K>{
	fn try_from(tensor:Tens<K::Elem>)->Result<Self,Self::Error>{
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
impl<B:Backend,K:BasicOps<B>,const N:usize> TryFrom<&View<K::Elem>> for Tensor<B,N,K>{
	fn try_from(tensor:&View<K::Elem>)->Result<Self,Self::Error>{
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
impl<B:Backend,K:BasicOps<B>,const N:usize> TryFrom<Tensor<B,N,K>> for BuiltinTensor<K::Elem>{
	fn try_from(value:Tensor<B,N,K>)->Result<Self,Self::Error>{
		let data=value.to_data();
		let layout=Layout::new(&data.shape);

		let data=data.to_vec()?;
		Ok(Tens::from_inner(data,layout).tensor())
	}
	type Error=DataError;
}
impl<B:Backend,K:BasicOps<B>,const N:usize> TryFrom<Tensor<B,N,K>> for Tens<K::Elem>{
	fn try_from(value:Tensor<B,N,K>)->Result<Self,Self::Error>{
		let data=value.to_data();
		let layout=Layout::new(&data.shape);

		let data=data.to_vec()?;
		Ok(Tens::from_inner(data,layout))
	}
	type Error=DataError;
}

#[cfg(feature="serial")]
/// deserialize a burn bool tensor from intertense tensor format with a specific component type bool not necessarily given by the kind
pub fn deserialize_bool_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Bool>,D::Error>{deserialize_elem_tensor::<B,D,bool,Bool,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn float tensor from intertense tensor format with a specific component type f64 not necessarily given by the kind
pub fn deserialize_double_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Float>,D::Error>{deserialize_elem_tensor::<B,D,f64,Float,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn tensor from intertense tensor format with a specific component type not necessarily given by the kind
pub fn deserialize_elem_tensor<'a,B:Backend,D:Deserializer<'a>,E:Deserialize<'a>+Element,K:BasicOps<B>,const N:usize>(deserializer:D)->Result<Tensor<B,N,K>,D::Error>{
	let tensor:Tens<E>=Tens::deserialize(deserializer)?;
	if tensor.rank()!=N{return Err(Derror::custom(format!("rank mismatch")))}

	let dims=tensor.dims().to_vec();

	let data=tensor.into_flat_vec();
	let device=&Default::default();
	let tensordata=TensorData::new(data,dims).convert::<K::Elem>();

	Ok(Tensor::from_data(tensordata, device))
}
#[cfg(feature="serial")]
/// deserialize a burn float tensor from intertense tensor format with a specific component type f32 not necessarily given by the kind
pub fn deserialize_float_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Float>,D::Error>{deserialize_elem_tensor::<B,D,f32,Float,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn int tensor from intertense tensor format with a specific component type i32 not necessarily given by the kind
pub fn deserialize_int_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Int>,D::Error>{deserialize_elem_tensor::<B,D,i32,Int,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn int tensor from intertense tensor format with a specific component type i64 not necessarily given by the kind
pub fn deserialize_long_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Int>,D::Error>{deserialize_elem_tensor::<B,D,i64,Int,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn tensor from intertense tensor format
pub fn deserialize_tensor<'a,B:Backend,D:Deserializer<'a>,K:BasicOps<B>,const N:usize>(deserializer:D)->Result<Tensor<B,N,K>,D::Error> where K::Elem:Deserialize<'a>{
	let tensor:Tens<K::Elem>=Tens::deserialize(deserializer)?;
	tensor.try_into().map_err(|e|Derror::custom(format!("{e:?}")))
}
#[cfg(feature="serial")]
/// deserialize a burn int tensor from intertense tensor format with a specific component type u32 not necessarily given by the kind
pub fn deserialize_uint_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Int>,D::Error>{deserialize_elem_tensor::<B,D,u32,Int,N>(deserializer)}
#[cfg(feature="serial")]
/// deserialize a burn int tensor from intertense tensor format with a specific component type u64 not necessarily given by the kind
pub fn deserialize_ulong_tensor<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Tensor<B,N,Int>,D::Error>{deserialize_elem_tensor::<B,D,u64,Int,N>(deserializer)}
#[cfg(feature="serial")]
/// serialize a burn bool tensor to intertense tensor format with a specific component type bool not necessarily given by the kind
pub fn serialize_bool_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Bool>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<B,bool,Bool,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn float tensor to intertense tensor format with a specific component type f64 not necessarily given by the kind
pub fn serialize_double_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Float>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<B,f64,Float,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn tensor to intertense tensor format with a specific component type not necessarily given by the kind
pub fn serialize_elem_tensor<B:Backend,E:Element+Serialize,K:BasicOps<B>,S:Serializer,const N:usize>(tensor:&Tensor<B,N,K>,serializer:S)->Result<S::Ok,S::Error>{
	let data=tensor.to_data();
	let layout=Layout::new(&data.shape);

	let data:Vec<E>=data.convert::<E>().to_vec().map_err(|e|Serror::custom(format!("{e:?}")))?;
	Tens::from_inner(data,layout).serialize(serializer)
}
#[cfg(feature="serial")]
/// serialize a burn float tensor to intertense tensor format with a specific component type f32 not necessarily given by the kind
pub fn serialize_float_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Float>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<B,f32,Float,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn int tensor to intertense tensor format with a specific component type i32 not necessarily given by the kind
pub fn serialize_int_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Int>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<B,i32,Int,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn int tensor to intertense tensor format with a specific component type i64 not necessarily given by the kind
pub fn serialize_long_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Int>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<B,i64,Int,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn tensor to intertense tensor format
pub fn serialize_tensor<B:Backend,K:BasicOps<B>,S:Serializer,const N:usize>(tensor:&Tensor<B,N,K>,serializer:S)->Result<S::Ok,S::Error> where K::Elem:Serialize{
	let tensor:Tens<K::Elem>=tensor.clone().try_into().map_err(|e|Serror::custom(format!("{e:?}")))?;
	tensor.serialize(serializer)
}
#[cfg(feature="serial")]
/// serialize a burn int tensor to intertense tensor format with a specific component type u32 not necessarily given by the kind
pub fn serialize_uint_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Int>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<B,u32,Int,S,N>(tensor,serializer)}
#[cfg(feature="serial")]
/// serialize a burn int tensor to intertense tensor format with a specific component type u64 not necessarily given by the kind
pub fn serialize_ulong_tensor<B:Backend,S:Serializer,const N:usize>(tensor:&Tensor<B,N,Int>,serializer:S)->Result<S::Ok,S::Error>{serialize_elem_tensor::<B,u64,Int,S,N>(tensor,serializer)}

use burn::{
	prelude::*,tensor::{BasicOps,DataError,Element}
};
use crate::builtin_tensor::{Error as TensorError,Layout,Tens,tensor::Tensor as BuiltinTensor,view::View};
#[cfg(feature="serial")]
use serde::{Deserialize,Deserializer,Serialize,Serializer,de::Error as Derror,ser::Error as Serror};

