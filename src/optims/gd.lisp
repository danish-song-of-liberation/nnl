(in-package :nnl.optims)

(defclass gd ()
  ((%parameters 
	:initform '()
	:accessor parameters
	:initarg :parameters
	:type list)
   (%learning-rate
	:initform 0.01
	:accessor learning-rate
	:initarg :lr
	:type real))
	
   (:documentation "todo"))
   
(defun update-parameters (parameters learning-rate)
  "todo optimize"
  
  (if (listp parameters)
    (mapcar #'(lambda (param) (update-parameters param learning-rate)) parameters)
    (nnl.math.autodiff:step! parameters :lr learning-rate)))   
   
(defmethod step ((self gd) &key (lr nil))
  (let ((self-lr (learning-rate self)))
    (when lr
      (setf self-lr lr))

    (update-parameters (parameters self) self-lr)))
		
(defmethod zero-grad ((self gd))
  (zero-parameters (parameters self)))	
