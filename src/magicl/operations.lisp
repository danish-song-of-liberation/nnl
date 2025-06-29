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
	
(defun sum (tensor &key axes)
  "The sum function is already available 
   in magicl, but it does not support 
   column or row summation and simply 
   returns a single sum. 
   
   This is crucial for broadcasting in 
   neural networks, so I have to implement 
   my own function."
   
  (let* ((shape (magicl:shape tensor))
		 (rank (length shape)))
		 
	(when axes
	  (dolist (axis axes)
	    (assert (< axis rank) () "~a Axis goes beyond the rank of the tensor (~a)." axis rank)))
		
	(cond
	  ((null axes)
	    (magicl:sum tensor))

	  ((= 1 (length axes))
	    (let* ((axis (first axes))
			   (new-shape (loop for i from 0 below rank unless (= i axis) collect (nth i shape)))
			   (result (magicl:zeros new-shape)))
			   
		   (if (= axis 0)
             (dotimes (j (second shape))
               (let ((sum 0))
                 (dotimes (i (first shape))
                   (setq sum (+ sum (magicl:tref tensor i j))))
				   
                 (setf (magicl:tref result j) sum)))
             
             (dotimes (i (first shape))
               (let ((sum 0))
                 (dotimes (j (second shape))
                   (setq sum (+ sum (magicl:tref tensor i j))))
				   
                 (setf (magicl:tref result i) sum))))
         
			  
		  result))
		  
	  (t 
	    (reduce #'(lambda (tensor axis) (sum tensor :axes (list axis))) axes :initial-value tensor)))))
	
(defun abs (tensor)	
  (magicl:map #'(lambda (x) (cl:abs x)) tensor))
  
(defun abs! (tensor)	
  (magicl:map! #'(lambda (x) (cl:abs x)) tensor))

(defun slice (tensor row-start row-end col-start col-end)
  "cause magicl:slice doesnt have normal docstring"
  
  (declare (type integer row-start row-end col-start col-end))
  
  (let* ((shape (magicl:shape tensor))
		 (all-rows (first shape))
		 (rows (1+ (- row-end row-start)))
		 (all-cols (second shape))
		 (cols (1+ (- col-end col-start)))
		 (matrix (magicl:make-tensor (get-magicl-type '(0 0) nnl.system::*calculus-system*) (list rows cols))))

	(declare (type list shape) (type integer all-rows rows all-cols cols))

	(assert (and (>= row-start 0) (< row-end all-rows) (>= col-start 0) (< col-end all-cols)) nil "todo description")

    (loop for i from 0 below rows
          for src-i from row-start do (loop for j from 0 below cols
											for src-j from col-start do (setf (magicl:tref matrix i j) (magicl:tref tensor src-i src-j))))
											
	matrix))

(defun trefv (tensor &rest indicies)
  (let* ((shape (magicl:shape tensor))
		 (last-dimension (car (last shape)))
		 (new-tensor (magicl:make-tensor (nnl.magicl:get-magicl-type '(0) nnl.system::*calculus-system*) (list last-dimension))))
	
	(dotimes (i last-dimension)
	  (setf (magicl:tref new-tensor i) (apply #'magicl:tref tensor (append indicies (list i)))))
	  
	new-tensor))
	
(defun trefm (tensor &rest indicies)
  (let* ((shape (magicl:shape tensor))
		 (prelast-dimensions (subseq shape 1))
		 (new-tensor (magicl:make-tensor (nnl.magicl:get-magicl-type '(0 0) nnl.system::*calculus-system*) prelast-dimensions)))
		 
	(dotimes (i (first prelast-dimensions))
	  (dotimes (j (second prelast-dimensions))
		(setf (magicl:tref new-tensor i j) (apply #'magicl:tref tensor (append indicies (list i j))))))
	
	new-tensor))
	