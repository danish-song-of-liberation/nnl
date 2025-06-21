(in-package :nnl.math)

#| Philipp Mainlender (1841 - 1876)
   - 
   Sie erschufen sich Götter, 
   weil sie es nicht konnten. |#

(defmethod ufunc ((self nnl.math.autodiff:tensor) &optional (funct #'+) (init 0.0))
  "todo"
  
  (let ((shapes (magicl:shape (nnl.math.autodiff::data self)))
		(accumulator init)
		(tensor (nnl.math.autodiff::data self)))
		
    (dotimes (i (car shapes))
	  (setq accumulator (funcall funct accumulator (magicl:tref tensor i))))
	  
	accumulator))
	
(defun make-tensor (data &key (requires-grad nil))
  (nnl.math.autodiff:make-tensor :data (nnl.magicl:coerce-to-tensor data) :requires-grad requires-grad))
	
(defun zeros (shape &key (requires-grad nil))
  (nnl.math.autodiff:make-tensor :data (magicl:zeros shape)	:requires-grad requires-grad))
  
(defun ones (shape &key (requires-grad nil))
  (nnl.math.autodiff:make-tensor :data (magicl:ones shape) :requires-grad requires-grad))
  
(defun full (shape filler &key (requires-grad nil))
  (let ((new-tensor (magicl:ones shape)))
    (magicl:scale! new-tensor filler)
	(nnl.math.autodiff:make-tensor :data new-tensor :requires-grad requires-grad)))
	
(defun arange (start-from end-to &optional (step 1) &key (requires-grad nil))
  (let ((new-tensor (magicl:make-tensor (nnl.magicl:get-magicl-type '(0) nnl.system::*calculus-system*) (list (/ end-to step)))))
    (loop for i from 0 below (/ end-to step)
		  for y from start-from below end-to
		  do (setf (magicl:tref new-tensor i) y)
		  finally (return (nnl.math.autodiff:make-tensor :data new-tensor :requires-grad requires-grad)))))
	
(defun linspace (start-from end-to elems &key (requires-grad nil) &aux (elems (1- elems)))
  (let ((new-tensor (magicl:make-tensor (nnl.magicl:get-magicl-type '(0) nnl.system::*calculus-system*) (list elems))))
    (loop for i from 0 below elems
		  for y from start-from to end-to by (/ end-to elems)
		  do (setf (magicl:tref new-tensor i) y)
		  finally (return (nnl.math.autodiff:make-tensor :data new-tensor :requires-grad requires-grad)))))
	
(defun eye (shape &key (requires-grad nil))
  (nnl.math.autodiff:make-tensor :data (magicl:eye shape) :requires-grad requires-grad))
  
(defun randn (shape &key (from -1.0) (to 1.0) (requires-grad nil))
  (nnl.math.autodiff:make-tensor :data (nnl.magicl:make-random-data shape :from from :to to) :requires-grad requires-grad))
	
(defun item (obj)
  (nnl.math.autodiff::data obj))

(defun gradient (obj)
  (nnl.math.autodiff::grad obj))

(defun grad (obj)
  (nnl.math.autodiff::grad obj))
  
(defmacro .+ (obj-1 obj-2)
  `(nnl.math.autodiff:+ ,obj-1 ,obj-2))	

(defmacro .- (obj-1 obj-2)
  `(nnl.math.autodiff:- ,obj-1 ,obj-2))	
    
(defmacro .* (obj-1 obj-2)
  `(nnl.math.autodiff:+ ,obj-1 ,obj-2))	
 
(defmacro matmul (obj-1 obj-2)
  `(nnl.math.autodiff:matmul ,obj-1 ,obj-2))	
   
(defmacro matvec (obj-1 obj-2)
  `(nnl.math.autodiff:matv ,obj-1 ,obj-2))	
   
(defmacro activation (obj funct)
  `(nnl.math.autodiff:activation ,obj ,funct))
  
(defmacro .relu (obj)
  `(activation ,obj #'relu))  
     
(defmacro .leaky-relu (obj)
  `(activation ,obj #'leaky-relu))    
 
(defmacro .sigmoid (obj)
  `(activation ,obj #'sigmoid))   
  
(defmacro .tanh (obj)
  `(activation ,obj #'tanh))    
  
(defmacro .linear (obj)
  `(activation ,obj #'linear)) 
  
(defun size (obj)
  (magicl:size (nnl.math.autodiff::data obj)))  
  
(defun order (obj)
  (magicl:order (nnl.math.autodiff::data obj)))    
  
(defun shape (obj)
  (magicl:shape (nnl.math.autodiff::data obj)))    
 
(defmacro at (obj &body indexs)
  `(magicl:tref (nnl.math.autodiff::data ,obj) ,@indexs))
  
(defmacro backprop (obj)
  `(nnl.math.autodiff:backprop ,obj))  
  
(defun transpose (obj)
  (let ((data-tensor (nnl.math.autodiff::data obj)))
    (setf (nnl.math.autodiff::data obj) (nnl.magicl:transpose (nnl.magicl:get-magicl-type (magicl:shape data-tensor) nnl.system::*calculus-system*) data-tensor)))
	
  obj)

(defun scale (obj multiplier)
  (magicl:scale! (nnl.math.autodiff::data obj) multiplier)
  
  obj)
  
(defmacro numerical (function tensor &key (epsilon 0.001) (precision 1))
  `(nnl.magicl:numerical ,function (nnl.math.autodiff::data ,tensor) :epsilon ,epsilon :precision ,precision))  
  