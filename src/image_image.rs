impl<C:Deref<Target=[P::Subpixel]>,P:Pixel> From<&ImageBuffer<P,C>> for Tensor<P::Subpixel>{
	fn from(image:&ImageBuffer<P,C>)->Self{
		let channels=P::CHANNEL_COUNT as usize;
		let height  =image.height()   as usize;
		let width   =image.width ()   as usize;

		let data:Vec<P::Subpixel>=image.pixels().flat_map(|p|p.channels().iter().copied()).collect();
		let dims=vec![height,width,channels];

		assert_eq!(channels*height*width,data.len());
		Tensor::new(data,dims)
	}
}
impl<C:Deref<Target=[P::Subpixel]>,P:Pixel> From<ImageBuffer<P,C>> for Tensor<P::Subpixel>{
	fn from(image:ImageBuffer<P,C>)->Self{Self::from(&image)}
}
impl<C:Deref<Target=[P::Subpixel]>+From<Vec<P::Subpixel>>,P:Pixel> TryFrom<&Tensor<P::Subpixel>> for ImageBuffer<P,C>{
	fn try_from(tensor:&Tensor<P::Subpixel>)->Result<Self,Error>{tensor.view().try_into()}
	type Error=Error;
}
impl<C:Deref<Target=[P::Subpixel]>+From<Vec<P::Subpixel>>,P:Pixel> TryFrom<Tensor<P::Subpixel>> for ImageBuffer<P,C>{
	fn try_from(tensor:Tensor<P::Subpixel>)->Result<Self,Error>{tensor.view().try_into()}
	type Error=Error;
}
impl<C:Deref<Target=[P::Subpixel]>+From<Vec<P::Subpixel>>,P:Pixel> TryFrom<&View<P::Subpixel>> for ImageBuffer<P,C>{
	fn try_from(tensor:&View<P::Subpixel>)->Result<Self,Error>{
		let channels=P::CHANNEL_COUNT as usize;
		let dims=tensor.dims();

		if dims.len()!=3       {return Err(Error::specific_rank_mismatch(tensor.get_layout(),3         ,"image"))}
		if dims[2]   !=channels{return Err(Error::specific_dim_mismatch (tensor.get_layout(),channels,2,"image"))}

		let height=dims[0] as u32;
		let width =dims[1] as u32;
		let v=tensor.flat_vec(None);

		let vl=v.len();
		let v=C::from(v);

		if v.len()<vl{return Err(Error::other_str("Image container type created from a vec big enough to store the image is expected to also be big enough",tensor.get_layout(),"image",None))}
		Ok(ImageBuffer::from_raw(width,height,v).expect("buffer size should be correct at this point"))
	}
	type Error=Error;
}

use crate::builtin_tensor::{error::Error,tensor::Tensor,View};
use image::{ImageBuffer,Pixel};
use std::ops::Deref;
