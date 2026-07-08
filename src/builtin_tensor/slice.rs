// slice pointer format: (ptr, meta)
// needed              : (flags, ptr, off, dims)
// we can forbid dim products greater than isize::MAX and interpret dims as a mixed radix number, with place digit counts given by the view at ptr
// for unification with std slices we need the ptr to be a E ptr, and for custom use of slice meta we neet the ptr to be a zst, but we should be able to cast between *E, *PhantomData<E>, and *Tensor<E> to meet different storage type requirements so long as std vs tensor is correctly tracked in metadata
// we can reduce dim bit count to 40 each because we probably won't be handling >1TB tensor slices
// reserve 4 bits for flags, need to condense 84 into 64.
// std slice don't need to track offset; we only need that so the ptr can actually be to a tensor struct that has the stride info
// flags: reserve 2, 2 remaining for 3 states: flat, scalar, tensor
// meta: 4 bits flags, 20 bits off hi, 40 bits dims
// ptr : just ptr if std, segment+6 bits off lo if tensor. segment has align 64
// segments every 2^26 bytes
const DIMOFF_BITS:usize=usize::BITS as usize*5/8;
const FLAGS_BITS:usize=4;
const FLAGS_SHIFT:usize=usize::BITS as usize-FLAGS_BITS;
const OFFH_BITS:usize=usize::BITS as usize-DIMOFF_BITS-FLAGS_BITS;
const OFFH_SHIFT:usize=DIMOFF_BITS;
const SEGMENT_INTERVAL_EXPONENT:usize=DIMOFF_BITS/2+6;
const UNPACKED_OFFL_MASK:usize=63;
const UNPACKED_OFFH_SHIFT:usize=6;
const SUBSEGMENT_MASK:usize=(1<<SEGMENT_INTERVAL_EXPONENT)-1;
const SUPERSEGMENT_MASK:usize=!SUBSEGMENT_MASK;

impl<E> Slice<E>{
	/// convert to slice parts
	fn into_slice_parts(&self)->SliceParts<*const E>{
		todo!()
	}


	/// get the slice ptr
	pub fn as_ptr(&self)->*const E{self.0.as_ptr() as *const E}
	/// get the metadata
	pub fn get_meta(&self)->usize{self.0.len()}
	/*/// create from raw parts. ptr should be acquired through
	pub fn from_raw_parts(ptr:*const E,meta:usize)->&Self{

	}*/
}
unsafe impl Send for ErasedSendSyncPtr{}
unsafe impl Sync for ErasedSendSyncPtr{}




/// regester a tensor segment for the tensor and offset, returning a pointer to the tensor segment, which has Segment<E> type but is cast to *mut E for type uniformity reasons
pub (crate) fn register_segment<E>(tensor:&Tensor<E>,offset:usize)->*mut E{
	let maphandle=SEGMENT_MAP.get_or_init(Default::default);
	let ptr=ErasedSendSyncPtr(tensor.as_ptr() as *mut ());

	maphandle.entry(ptr).or_insert_with(||unsafe{
		let mut segment=tensor.clone_ref().transmute_components();
		segment.offset(offset&SUPERSEGMENT_MASK);

		Box::pin(Segment(segment))
	}).0.as_ptr() as *mut E
}

/// map from E pointers to tensor segments with the actual pointers
static SEGMENT_MAP:OnceLock<DashMap<ErasedSendSyncPtr,Pin<Box<Segment<()>>>>>=OnceLock::new();

#[repr(C)]
#[repr(align(64))]
/// store a "tensor" as a sort of offset checkpoint, since we don't have enough metadata per pointer to fully explore the address space otherwise
struct Segment<E>(Tensor<E>);	// vc safety: >1

#[derive(Clone,Copy,Eq,Hash,PartialEq)]
/// pointer for potentially cross thread type erased activities
struct ErasedSendSyncPtr(*mut ());
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// transient represention of the slice metadata
pub struct Meta{dims:[u8;5],flags:u8,off:[u8;5]}
#[repr(transparent)]
/// DST to store the slice pointer and metadata.
pub struct Slice<E>([PhantomData<E>]);
#[repr(u8)]
/// the three kinds of slices...
enum SliceKind{Flat,Segment,Scalar}
/// alternative internal representation. use dims for len when flat
struct SliceParts<P>{kind:SliceKind,dims:u32,offset:u32,ptr:P}

use dashmap::DashMap;
use std::{
	marker::PhantomData,pin::Pin,ptr,sync::OnceLock
};
use super::Tensor;
