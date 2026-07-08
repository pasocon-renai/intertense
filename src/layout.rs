impl AsMut<Layout> for Layout{
	fn as_mut(&mut self)->&mut Layout{self}
}
impl AsRef<Layout> for Layout{
	fn as_ref(&self)->&Layout{self}
}
impl Layout{
	/// computes the offset of a component given its coordinates
	pub fn compute_offset<I:SignedIndexPosition>(&self,indices:&[I])->usize{position::compute_offset(self.dims(),indices,self.strides())}
	/// counts the number of components. invalid layouts may return something unhelpful or unexpected
	pub fn count(&self)->usize{self.dims().iter().product()}
	/// references the dims
	pub fn dims(&self)->&Vec<usize>{&self.inner.0}
	/// references the dims. note that most nontrivial functionality expects the lengths of dims and strides to be equal
	pub fn dims_mut(&mut self)->&mut Vec<usize>{&mut Arc::make_mut(&mut self.inner).0}
	/// creates a layout from the dims and strides. does not check validity
	pub fn from_inner(dims:Vec<usize>,strides:Vec<isize>)->Self{
		Self{inner:Arc::from((dims,strides))}
	}
	/// get the dimension of the corresponding index. panics if out of bounds of the dims vector
	pub fn get_dim(&self,index:impl SignedIndexPosition)->usize{
		let index=index.try_into().unwrap_or(isize::MIN);
		let index=position::unsign_index(index,self.dims().len()).unwrap();

		self.dims()[index]
	}
	/// get the stride of the corresponding index. panics if out of bounds of the strides vector
	pub fn get_stride(&self,index:impl SignedIndexPosition)->isize{
		let index=index.try_into().unwrap_or(isize::MIN);
		let index=position::unsign_index(index,self.strides().len()).unwrap();

		self.strides()[index]
	}
	/// references the dims. note that most nontrivial functionality expects the lengths of dims and strides to be equal
	pub fn inner_mut(&mut self)->(&mut Vec<usize>,&mut Vec<isize>){
		let (d,s)=Arc::make_mut(&mut self.inner);
		(d,s)
	}
	/// converts into the dims
	pub fn into_dims(self)->Vec<usize>{
		if Arc::strong_count(&self.inner)==1{
			Arc::into_inner(self.inner).unwrap().0
		}else{
			self.dims().to_vec()
		}
	}
	/// convert into the inner value
	pub fn into_inner(self)->(Vec<usize>,Vec<isize>){
		if Arc::strong_count(&self.inner)==1{
			Arc::into_inner(self.inner).unwrap()
		}else{
			(self.dims().to_vec(),self.strides().to_vec())
		}
	}
	/// checks if the dims contain 0
	pub fn is_empty(&self)->bool{self.dims().contains(&0)}
	/// checks if the layout is normalized, returning true if it is and false if it isn't. also returns false if the layout is invalid. A layout is considered normalized if and only if mapping a lexiconographic iteration over the indices to components would iterate over the same components in the same order as iterating over the underlying buffer. Normalizing a layout may require normalizing the buffer too.
	pub fn is_normalized(&self)->bool{
		let mut acc:usize=1;
		if self.dims().len()!=self.strides().len(){return false}
									// the normalization condition generally equivalent to whether each stride is equal to the product of all the dims after it
		for (&d,&s) in self.dims().iter().rev().zip(self.strides().iter().rev()){
			if acc as isize!=s{return false}
			acc=acc.saturating_mul(d);

			if acc>isize::MAX as usize{return false}
		}
		true
	}
	/// check if this layout represents a scalar. We use dims.len()==0 to represent scalars
	pub fn is_scalar(&self)->bool{self.dims().is_empty()}
	/// computes the length of a slice required to store a tensor with this layout. the backing vec of a tensor will store the data in a slice of this length starting at index offset in a vec of at least offset+len. May return something unexpected or unhelpful if the layout is invalid
	pub fn len(&self)->usize{position::buffer_len(self.dims(),self.strides())}
	/// creates a new tensor layout from the dimensions. the dim product probably shouldn't exceed isize::MAX, but this function doesn't check
	pub fn new<D:AsRef<[usize]>>(dims:D)->Self{
		match Self::try_new(dims){
			Err(e)=>panic!("{e}"),
			Ok(x)=>x
		}
	}
	/// returns the number of dims in the tensor
	pub fn rank(&self)->usize{self.dims().len()}
	/// references the strides
	pub fn strides(&self)->&Vec<isize>{&self.inner.1}
	/// references the strides. note that most nontrivial functionality expects the lengths of dims and strides to be equal
	pub fn strides_mut(&mut self)->&mut Vec<isize>{&mut Arc::make_mut(&mut self.inner).1}
	/// try to create a new layout, returning an error if the dims are invalid
	pub fn try_new<D:AsRef<[usize]>>(dims:D)->Result<Self>{
		let dims=dims.as_ref();
		error::check_dims(dims).map_err(|e|e.with_op("new"))?;

		let mut acc:usize=1;
		let dims=dims.to_vec();
		let rank=dims.len();
		let mut strides=vec![0;rank];

		for ix in (0..rank).rev(){
			strides[ix]=acc as isize;
			acc*=dims[ix];
		}
		Ok(Self::from_inner(dims,strides))
	}
	/// checks if a layout has a valid combination of dims and strides
	/// this function verifies:
	/// count does not overflow usize
	/// dims do not overflow isize
	/// the required buffer len does not overflow isize
	/// the required buffer len is within bufferlen
	pub fn validate(&self,bufferlen:usize)->Result<()>{
		error::check_layout(self.dims(),self.strides()).map_err(|e|e.with_op("validate"))?;
		let len=self.len();

		if bufferlen<len{return Err(Error::buffer_too_small(self.clone(),bufferlen,len,"validate"))}
		Ok(())
	}
	/// check if a layout has a valid combination of dims and strides and won't alias its components. Rather than enumerating tensor positions, it checks axis's strides and detects overlap using the least common multiple of stride magnitudes.
	/// this function verifies:
	/// count does not overflow usize
	/// dims do not overflow isize
	/// the required buffer len does not overflow isize
	/// the required buffer len is within bufferlen
	/// no two positions would map to the same component offset within the tensor buffer
	pub fn validate_mut(&self,bufferlen:usize)->Result<()>{
		error::check_layout_mut(self.dims(),self.strides()).map_err(|e|e.with_layout(self.clone()).with_op("validate"))?;
		let len=self.len();

		if bufferlen<len{return Err(Error::buffer_too_small(self.clone(),bufferlen,len,"validate"))}
		Ok(())
	}

	/*/// computes the number of components in a tensor with this lauout. errors if the layout is invalid
	pub fn try_count(&self)->Result<usize>{self.check_validity().map(|_|self.count()).map_err(|e|e.with_op("get_count"))}
	/// creates a layout from the dims and strides, checking that validity is possible
	pub fn try_from_inner(dims:Vec<usize>,strides:Vec<isize>)->Result<Self>{
		let result=Self{inner:Arc::from((dims,strides))};
		result.check_validity().map_err(|e|e.with_op("from_inner"))?;

		Ok(result)
	}
	/// get the dimension of the corresponding index
	pub fn try_get_dim<I:TryInto<isize>>(&self,index:I)->Result<usize> where I::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(Box::new(e),self.clone(),"get_dim",None))?;
		let rank=self.dims().len();

		let index=position::normalize_index(rank,index).ok_or(Error::invalid_index(index,self.clone(),"get_dim"))?;
		Ok(self.dims()[index])
	}
	/// get the stride of the corresponding index
	pub fn try_get_stride<I:TryInto<isize>>(&self,index:I)->Result<isize> where I::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(Box::new(e),self.clone(),"get_stride",None))?;
		let rank=self.strides().len();

		let index=position::normalize_index(rank,index).ok_or(Error::invalid_index(self.clone(),index,"get_stride"))?;
		Ok(self.strides()[index])
	}
	/// computes the length of a slice required to store a tensor with this layout. the backing vec of a tensor will store the data in a slice of this length starting at index offset in a vec of at least offset+len. errors if the layout is invalid
	pub fn try_len(&self)->Result<usize>{self.check_validity().map(|_|self.len()).map_err(|e|e.with_op("get_len"))}
	/// creates a new tensor layout from the dimensions. their product should not exceed isize::MAX
	pub fn try_new<D:AsRef<[usize]>>(dims:D)->Result<Self>{
		let dims=dims.as_ref().to_vec();
		let rank=dims.len();

		let mut acc:Option<usize>=Some(1);
		let mut strides=vec![0;rank];

		for n in (0..rank).rev(){
			strides[n]=acc.and_then(|x|x.try_into().ok()).unwrap_or(isize::MIN);
			acc       =acc.and_then(|x|dims[n].checked_mul(x));
		}
		let result=Self{inner:Arc::new((dims,strides))};

		if let Some(a)=acc&&a<isize::MAX as usize{
			Ok(result)
		}else{
			Err(Error::too_big(None,acc,result,acc,"new"))
		}
	}*/
}

#[cfg(feature="serial")]
mod serial{
	impl<'a> Deserialize<'a> for L{
		fn deserialize<D:Deserializer<'a>>(deserializer:D)->StdResult<Self,D::Error>{
			let l=Layout::deserialize(deserializer)?;
			Ok(L::from_inner(l.dims,l.strides))
		}
	}
	impl Serialize for L{
		fn serialize<S:Serializer>(&self,serializer:S)->StdResult<S::Ok,S::Error>{
			let (dims,strides)=self.clone().into_inner();
			let stray=false;

			Layout{dims,stray,strides}.serialize(serializer)
		}
	}

	#[derive(Deserialize,Serialize)]
	/// serial layout structure to ensure compatibility with legacy files
	struct Layout{dims:Vec<usize>,stray:bool,strides:Vec<isize>}

	use serde::{Deserialize,Deserializer,Serialize,Serializer};
	use std::result::Result as StdResult;
	use super::Layout as L;
}

#[derive(Clone,Debug,Default,Eq,Hash,PartialEq)]
/// dimensions and strides for how a tensor data is layed out in memory
pub struct Layout{inner:Arc<(Vec<usize>,Vec<isize>)>}

use std::sync::Arc;
use super::{
	Error,Result,error,position::{SignedIndexPosition,self}
};
