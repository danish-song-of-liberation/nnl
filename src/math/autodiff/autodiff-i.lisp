(in-package :nnl.math.autodiff)

(defclass tensor ()
  ((%data 
	:initform nil
	:accessor data
	:initarg :data
	:documentation "Tensor that stores computational data.
					The type is not specified due to the separation of data types.
					type = magicl:tensor (or matrix/vector)/single-float (or double-float)")
   (%grad 
	:initform 0.0
	:accessor grad
	:documentation "Gradient of the tensor
					The type is not specified due to the separation of data types.
					type = magicl:tensor (or matrix/vector)/single-float (or double-float)
					usually 0.0 cause needed for backprop")
   (%parents 
    :initform '() 	
	:accessor parents
	:initarg :parents
	:type list
	:documentation "Tensor parents for backpropagation")
   (%requires-grad 
    :initform nil
	:reader requires-grad
	:initarg :requires-grad
	:type boolean
	:documentation "in the case of nil, gradients will not be calculated. in the case of t, there will be")
   (%backward 
    :initform #'(lambda ())
	:accessor backward
	:type function
	:documentation "function for backpropagation")))

(defmethod intern-add ((self tensor) other)
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:.+ (data self) (data other))
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list self other))))
	
	(setf (backward out) #'(lambda () (derivative-add out self other))) 
	
	out))
	
(defmethod intern-sub ((self tensor) other)
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:.- (data self) (data other))
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list self other))))
	
	(setf (backward out) #'(lambda () (derivative-sub out self other))) 
	
	out))
	
(defmethod intern-sub (other (self tensor))
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:.- (data other) (data self))
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list other self))))
	
	(setf (backward out) #'(lambda () (derivative-sub out other self))) 
	
	out))
	
(defmethod intern-mul ((self tensor) other)
  (let* ((reccurent-p (eq self other))
         (other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:.* (data other) (data self))
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list other self))))
	
	(setf (backward out) #'(lambda () (derivative-mul out other self reccurent-p))) 
	
	out))
	
(defmethod matv ((self tensor) other)
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:@ (data self) (data other))
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list self other))))
	
	(setf (backward out) #'(lambda () (derivative-matv out self other))) 
	
	out))	
	
(defun build-topo (v topo visited)
  (if (member v visited)
    (values topo visited)
    (let ((new-visited (cons v visited)))
	  (dolist (parent (parents v))
	    (multiple-value-setq (topo new-visited)
		  (build-topo parent topo new-visited)))
	  (values (cons v topo) new-visited))))
				
(defmethod backprop ((self tensor) &key (init-grad 1.0))
  (let ((type (nnl.magicl:get-magicl-type (magicl:shape (data self)) nnl.system::*calculus-system*)))		
	(multiple-value-bind (topo visited) (build-topo self '() '())
	  (setf (grad self) (magicl:make-tensor type (magicl:shape (data self)) :initial-element init-grad))
	  
	  (dolist (node topo)
	    (funcall (backward node))))))
				