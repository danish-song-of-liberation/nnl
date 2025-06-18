(in-package :nnl.magicl)

(defun get-magicl-type (dimensions type)
  (cond ((= (length dimensions) 1)
           (ecase type
             (single-float 'magicl:vector/single-float)
             (double-float 'magicl:vector/double-float)))
        ((= (length dimensions) 2)
           (ecase type
             (single-float 'magicl:matrix/single-float)
             (double-float 'magicl:matrix/double-float)))
        (t (ecase type
             (single-float 'magicl:tensor/single-float)
             (double-float 'magicl:tensor/double-float)))))
				 
(defun get-magicl-type! (tensor)
  (get-magicl-type (magicl:shape tensor) nnl.system::*calculus-system*))

(defun coerce-array-to-tensor (data)
  (declare (type array data))
  (assert (typep data 'array))
  
  (let* ((dims (array-dimensions data))
		 (shapes (length dims)))
		
    (declare (type fixnum shapes))
		
    (case shapes
	  (1 (coerce-vector-to-magicl-vector data (get-magicl-type dims nnl.system::*calculus-system*) dims))
	  (2 (coerce-matrix-to-magicl-matrix data (get-magicl-type dims nnl.system::*calculus-system*) dims))
	  (3 (coerce-tensor-to-magicl-tensor data (get-magicl-type dims nnl.system::*calculus-system*) dims)))))

(defun coerce-list-to-tensor (data)
  "todo"
  
  )

(defun coerce-to-tensor (data)
  (typecase data
    (array (coerce-array-to-tensor data))
	(list (coerce-list-to-tensor data))))

(defun make-random-data (shapes &key (from 0.0) (to 1.0))
  (case (length shapes)
    (1 (make-random-vector (get-magicl-type shapes nnl.system::*calculus-system*) shapes from to))
	(2 (make-random-matrix (get-magicl-type shapes nnl.system::*calculus-system*) shapes from to))
	(3 (make-random-tensor (get-magicl-type shapes nnl.system::*calculus-system*) shapes from to))))
