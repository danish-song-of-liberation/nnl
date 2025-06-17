(in-package :nnl.magicl)

(defun coerce-vector-to-magicl-vector (data data-type data-size)
  "Low-level function that converts a Common Lisp array (specifically, a vector)
   into a Magicl tensor-vector.
   
   It handles coercion based on the specified data type (SINGLE-FLOAT or DOUBLE-FLOAT)"

  (declare (type (array *) data) 
		   (type list data-size)
		   (type symbol data-type))
  
  (let ((vector (magicl:make-tensor data-type data-size)))				
    (dotimes (it (car data-size))
	  (setf (magicl:tref vector it) (aref data it)))
	  
	vector))

(defun coerce-matrix-to-magicl-matrix (data data-type shapes)
  "Low-level function that converts a Common Lisp array (specifically, a matrix)
   into a Magicl tensor-matrix.
   
   It handles coercion based on the specified data type (SINGLE-FLOAT or DOUBLE-FLOAT)"

  (declare (type (array * *) data) 
		   (type list shapes)
		   (type symbol data-type))
  
  (let* ((matrix (magicl:make-tensor data-type shapes)))
	(dotimes (i (car shapes))
	  (dotimes (j (cadr shapes))
	    (setf (magicl:tref matrix i j) (aref data i j))))
		
	matrix))
	
(defun coerce-tensor-to-magicl-tensor (data data-type shapes)
  "Low-level(ah how I like to copy paste my own code) function that converts a Common Lisp array (specifically, a matrix)
   into a Magicl tensor-matrix.
   
   It handles coercion based on the specified data type (SINGLE-FLOAT or DOUBLE-FLOAT)"
  (declare (type array data)
		   (type list shapes)
		   (type symbol data-type))
  
  (let* ((tensor (magicl:make-tensor data-type shapes)))
    (dotimes (i (car shapes))
      (dotimes (j (cadr shapes))
		(dotimes (k (caddr shapes))
		  (setf (magicl:tref tensor i j k) (aref data i j k)))))
		
	tensor)) 	
	