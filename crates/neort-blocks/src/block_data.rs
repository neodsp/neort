use core::marker::PhantomData;

#[cfg(feature = "alloc")]
use alloc::alloc::{dealloc, Layout};

use crate::Num;

pub trait BlockDataConst {
    type Num: Num;
    fn capacity(&self) -> usize;
    fn as_ptr(&self) -> *const Self::Num;
}

pub trait BlockDataMut: BlockDataConst {
    fn as_mut_ptr(&mut self) -> *mut Self::Num;
}

pub struct Stack<T, const CAPACITY: usize> {
    pub(crate) data: [T; CAPACITY],
}

#[cfg(feature = "alloc")]
pub struct Heap<T> {
    pub(crate) ptr: *mut T,
    pub(crate) capacity: usize,
}

pub struct View<'a, T> {
    pub(crate) ptr: *const T,
    pub(crate) capacity: usize,
    pub(crate) _phantom: PhantomData<&'a T>,
}

pub struct ViewMut<'a, T> {
    pub(crate) ptr: *mut T,
    pub(crate) capacity: usize,
    pub(crate) _phantom: PhantomData<&'a mut T>,
}

impl<T: Num, const CAPACITY: usize> BlockDataConst for Stack<T, CAPACITY> {
    type Num = T;

    #[inline]
    fn capacity(&self) -> usize {
        CAPACITY
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
}

impl<T: Num, const CAPACITY: usize> BlockDataMut for Stack<T, CAPACITY> {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

#[cfg(feature = "alloc")]
impl<T: Num> BlockDataConst for Heap<T> {
    type Num = T;

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

#[cfg(feature = "alloc")]
impl<T: Num> BlockDataMut for Heap<T> {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

#[cfg(feature = "alloc")]
impl<T> Drop for Heap<T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::array::<T>(self.capacity).unwrap();
            dealloc(self.ptr as *mut u8, layout);
        }
    }
}

impl<T: Num> BlockDataConst for View<'_, T> {
    type Num = T;

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<T: Num> BlockDataConst for ViewMut<'_, T> {
    type Num = T;

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<T: Num> BlockDataMut for ViewMut<'_, T> {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
