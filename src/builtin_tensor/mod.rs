// tensor operation errors
pub mod error;
// indexing utilities
pub mod position;
// buffer layout utilities
pub mod layout;
// slice like slicing
//pub mod slice;
// owned tensor
//pub mod tensor;
/// borrowed or transient tensor views
pub mod view;

pub use {
	error::{Error,Result},position::{PositionIter,Position},layout::Layout,//tensor::Tensor,view::{View,ViewMut,ViewRef}
};
