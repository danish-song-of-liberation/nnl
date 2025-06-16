(in-package :nnl.magicl)

(defun coerce-array-to-tensor (data)
  (declare (type array data))
  (assert (typep data 'array))
  
  (let* ((dims (array-dimensions data))
		 (shapes (length dims)))
		
    (declare (type fixnum shapes))
		
    (case shapes
	  (1 (coerce-vector-to-magicl-vector data nnl.system::*calculus-system* dims))
	  (2 (coerce-matrix-to-magicl-matrix data nnl.system::*calculus-system* dims))
	  (3 (coerce-tensor-to-magicl-tensor data nnl.system::*calculus-system* dims)))))
