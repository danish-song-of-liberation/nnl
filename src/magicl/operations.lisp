(in-package :nnl.magicl)

(declaim (inline transpose))
(declaim (inline outer))

(defun transpose (data-type matrix)
  (declare (type symbol data-type))
  
  (let* ((shapes (magicl:shape matrix))
		 (new-matrix (magicl:make-tensor data-type (reverse shapes))))
		 
	(declare (type list shapes))
		
    (dotimes (i (first shapes))
	  (dotimes (j (second shapes))
	    (setf (magicl:tref new-matrix j i) (magicl:tref matrix i j))))
		
	new-matrix))
	
(defun outer (data-type vector-1 vector-2)
  (declare (type symbol data-type))
  
  (let* ((shape-1 (the fixnum (car (magicl:shape vector-1))))
		 (shape-2 (the fixnum (car (magicl:shape vector-2))))
		 (new-matrix (magicl:make-tensor data-type (list shape-1 shape-2))))
		
	(declare (type fixnum shape-1 shape-2))
	(assert (= shape-1 shape-2))
	
	(dotimes (i shape-1)
	  (dotimes (j shape-2)
	    (setf (magicl:tref new-matrix i j) (* (magicl:tref vector-1 i) (magicl:tref vector-2 j)))))
	
	new-matrix))
	
	
		