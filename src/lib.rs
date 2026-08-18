impl<I:Display,O:Display> Display for ConversionError<I,O>{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		match self{Self::Input(e)=>e.fmt(f),Self::Output(e)=>e.fmt(f)}
	}
}
impl<I:Error,O:Error> Error for ConversionError<I,O>{}
#[derive(Clone,Copy,Debug,Eq,Hash,PartialEq)]
pub enum ConversionError<I,O>{Input(I),Output(O)}

//#[cfg(any(feature="match-tensor",feature="umya-sheet"))]
// functionality in common with excel and matching related features, named after the xmatch function
//mod xmatch;
/// Builtin tensor functionality.
/// For this library's purposes...
/// An **axis** is an abstract geometric direction in which tensor components may be arranged. For consistency with common tensor operation names, _dim rather than _axis suffix may be applied to function names that apply to a specific axis
/// A **count** is the total number of components in a tensor, not necessarily the same as its length. Count can be any value of usize, though particularly large counts might be not useful due to only being achievable through broadcasting views that are never realized.
/// A **dimension** (dim) a tensor's extent along a particular axis. Dims should not exceed isize::MAX, and their product should not exceed usize::MAX.
/// An **index** (ix) is a signed integer selecting an axis.
/// A **layout** is a tensor's dimensions and strides, usually stored in a Layout. It doesn't include offsets, which are handled separately at a view specific level since owned tensors are eagerly trimmed. Layouts have 2 varying degrees of validity. A layout may have either no validity (invalid), shared validity (shared-valid), or mutable validity (mut-valid), with the difference between shared-valid and mut-valid being that mut-valid layouts are not allowed to have multiple positions refer to the same component. A layout is considered invalid if any of its dims exceed isize::MAX, if their product exceeds usize::MAX, if its buffer length overflows isize, or if its dims and strides have mismatched ranks. Layouts that are not invalid are considered 'valid', however, a valid layout is only 'valid for' buffer lengths greater than the greatest component offset of a tensor with that layout.
/// A **length** (len) is the length in-use of a tensor's buffer, not necessarily the same its count.
/// A **position** (px) tells the location of a component along one or more axes. Generally has type Position if possibly more than one, type isize if known at coding time to be only one.
/// A **coordinate** is a singular position.
/// A **rank** is the number of axes in a tensor.
/// A **view** is tensor described by it's own layout and a buffer pointer from another preexisting tensor
/// A list of ranges of coordinates refers to the slice of components whose positions along each axis are within the corresponding range. A range of lists of coordinates refers to the components at positions between the range bounds in an iteration last-axis-fastest ordered with respect to position
pub mod builtin_tensor;
#[cfg(feature="burn-ml")]
/// machine learning interop with burn
pub mod burn_ml;
#[cfg(feature="image-image")]
pub mod image_image;
#[cfg(any(feature="match-tensor",feature="umya-sheet"))]
/// builtin tensor matching functionality
pub mod match_tensor;
#[cfg(feature="nd-array")]
/// ndarray interop
pub mod nd_array;
#[cfg(feature="umya-sheet")]
///excel-like spreadsheet ops
pub mod sheet_ops;
//#[cfg(feature="serial")]
// serde as a tensor
//pub mod serde_tensor;
#[cfg(feature="umya-sheet")]
/// excel interop with umya spreadsheet
pub mod umya_sheet;

#[track_caller]
/// converts between two tensor like formats using the builtin tensor as an intermediate
pub fn convert<E,T:TryInto<Tensor<E>>,U:TryFrom<Tensor<E>>>(tensor:T)->U where T::Error:Display,U::Error:Display{builtin_tensor::error::unwrap_or_panic(try_convert(tensor))}
/// converts between two tensor like formats using the builtin tensor as an intermediate
pub fn try_convert<E,T:TryInto<Tensor<E>>,U:TryFrom<Tensor<E>>>(tensor:T)->Result<U,ConversionError<T::Error,U::Error>>{U::try_from(tensor.try_into().map_err(ConversionError::Input)?).map_err(ConversionError::Output)}

use builtin_tensor::tensor::Tensor;
use std::{
	error::Error,fmt::{Display,Formatter,Result as FmtResult}
};
