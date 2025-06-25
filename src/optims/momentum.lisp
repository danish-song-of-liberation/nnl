(in-package :nnl.optims)

; TODO

(defclass momentum ()
  ((%parameters 
	:initform '()
	:accessor parameters
	:initarg :parameters
	:type list)
   (%learning-rate
	:initform 0.01
	:accessor learning-rate
	:initarg :lr
	:type real)
   (%momentum
    :initform 0.9
    :accessor momentum
    :initarg :momentum
    :type real)
   (%velocity
    :initform '()
    :accessor velocity))
	
   (:documentation "todo"))
   
(defun init-velocity (params)
  (if (listp params)
    (mapcar #'init-velocity params)
	(nnl.math:zeros-like params)))
   
(defmethod initialize-instance :after ((self momentum) &key)
  (setf (velocity self) (init-velocity (parameters self))))   
   
(defun update-parameters-momentum (parameters velocity learning-rate momentum)
  "todo optimize"
  
  (if (listp parameters)
    (mapcar #'(lambda (param v) (update-parameters-momentum param v learning-rate momentum)) parameters velocity)
	
    (progn
	  (let* ((grad (nnl.math:grad parameters))
		     (new-velocity (nnl.math:+ (nnl.math:scale velocity momentum) (magicl:scale grad (- learning-rate))))
			 (new-params (nnl.math:+ parameters new-velocity)))
			 
		(setf (nnl.math.autodiff::data parameters) (nnl.math.autodiff::data new-params))	 
			 
		new-velocity))))
		
(defmethod step ((self momentum) &key (lr nil))
  (let ((self-lr (learning-rate self)))
    (when lr
      (setf self-lr lr))

    (let ((new-velocity (update-parameters-momentum (parameters self) (velocity self) self-lr (momentum self))))
	  (setf (velocity self) new-velocity))))
		
(defmethod zero-grad ((self momentum))
  (zero-parameters (parameters self)))	
