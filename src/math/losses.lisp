(in-package :nnl.math.autodiff)

(defun derivative-mse (out self other)
  (when (requires-grad self)
    (setf (grad self) (magicl:.+ (grad self) (magicl:scale (magicl:.- (data self) (data other)) 2)))))

(defmethod mse ((self tensor) other)
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (magicl:.^ (magicl:.- (data self) (data other)) 2)
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list self other))))
	
	(setf (backward out) #'(lambda () (derivative-mse out self other))) 
	
	out))
	
(defun derivative-mae (out self other)
  (when (requires-grad self)
    (let ((diff (magicl:.- (data self) (data other))))
      (setf (grad self) (magicl:.+ (grad self) (magicl:.* (magicl:map #'signum diff) (grad out)))))))
	
(defmethod mae ((self tensor) other)	
  (let* ((other (if (typep other 'tensor) other (make-instance 'tensor :data other)))
		 (out (make-instance 'tensor 
				 :data (nnl.magicl:abs (magicl:.- (data self) (data other)))
				 :requires-grad (or (requires-grad self) (requires-grad other))
				 :parents (list self other))))
	
	(setf (backward out) #'(lambda () (derivative-mae out self other))) 
	
	out))
		