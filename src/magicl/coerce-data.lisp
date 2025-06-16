(in-package :nnl.magicl)

(defun coerce-vector-to-magicl-vector (data data-type data-size)
  "Low-level function that converts a Common Lisp array (specifically, a vector)
   into a Magicl tensor-vector.
   
   It handles coercion based on the specified data type (SINGLE-FLOAT or DOUBLE-FLOAT)"

  (declare (type (array *) data) 
		   (type list data-size)
		   (type symbol data-type))
  
  (let* ((type-for-vector (the symbol (ecase data-type
										(single-float 'magicl:vector/single-float)
										(double-float 'magicl:vector/double-float)
										(otherwise (error "~%~a: Unsupported data type: ~a. Supported types are SINGLE-FLOAT and DOUBLE-FLOAT.~%" (function coerce-vector-to-vector-magicl) data-type)))))
		 (vector (magicl:make-tensor type-for-vector data-size)))
						
	(declare (type symbol type-for-vector))					
	
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
  
  (let* ((type-for-matrix (the symbol (ecase data-type
										(single-float 'magicl:matrix/single-float)
										(double-float 'magicl:matrix/double-float)
										(otherwise (error "~%~a: Unsupported data type: ~a. Supported types are SINGLE-FLOAT and DOUBLE-FLOAT.~%" (function coerce-matrix-to-matrix-magicl) data-type)))))
		 (matrix (magicl:make-tensor type-for-matrix shapes)))
    
	(declare (type symbol type-for-matrix)) 
	
	(dotimes (i (car shapes))
	  (dotimes (j (cadr shapes))
	    (setf (magicl:tref matrix i j) (aref data i j))))
		
	matrix))
	
(defun coerce-tensor-to-magicl-tensor (data data-type shapes)
  "Low-level(ah how I like to copy paste my own code) function that converts a Common Lisp array (specifically, a matrix)
   into a Magicl tensor-matrix.
   
   It handles coercion based on the specified data type (SINGLE-FLOAT or DOUBLE-FLOAT)"

  (declare (type (array * * *) data) 
		   (type list shapes)
		   (type symbol data-type))
  
  (let* ((type-for-tensor (the symbol (ecase data-type
										(single-float 'magicl:tensor/single-float)
										(double-float 'magicl:tensor/double-float)
										(otherwise (error "~%~a: Unsupported data type: ~a. Supported types are SINGLE-FLOAT and DOUBLE-FLOAT.~%" (function coerce-matrix-to-matrix-magicl) data-type)))))
		 (tensor (magicl:make-tensor type-for-tensor shapes)))
    
	(declare (type symbol type-for-matrix))

    (dotimes (i (car shapes))
      (dotimes (j (cadr shapes))
		(dotimes (k (caddr shapes))
		  (setf (magicl:tref tensor i j k) (aref data i j k)))))
		
	tensor)) 	
	