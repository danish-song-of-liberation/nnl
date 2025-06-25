(in-package :nnl.math.autodiff)

(defmethod hstack ((self tensor) other)
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:hstack (list (data self) (data other)))
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list self other))))
	
	(setf (backward out) #'(lambda () (derivative-hstack out self other))) 
	
	out))
