(in-package :nnl.math.autodiff)

(defmethod intern-hstack ((self tensor) other)
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:hstack (list (data self) (data other)))
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list self other))))
	
	(setf (backward out) #'(lambda () (derivative-hstack out self other))) 
	
	out))

(defmethod intern-hstack (other (self tensor))
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:hstack (list (data other) (data self)))
				 :requires-grad (or (requires-grad other) (requires-grad self))
				 :parents (list other self))))
	
	(setf (backward out) #'(lambda () (derivative-hstack out other self))) 
	
	out))

(defmethod intern-vstack ((self tensor) other)
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:vstack (list (data self) (data other)))
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list self other))))
	
	(setf (backward out) #'(lambda () (derivative-vstack out self other))) 
	
	out))
	
(defmethod intern-vstack (other (self tensor))
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:vstack (list (data other) (data self)))
				 :requires-grad (or (requires-grad other) (requires-grad self))
				 :parents (list other self))))
	
	(setf (backward out) #'(lambda () (derivative-vstack out other self))) 
	
	out))
		
	