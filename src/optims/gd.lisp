(in-package :nnl.optims)

(defclass intern-gradient-descent ()
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
   
(defmethod backpropagation ((self intern-gradient-descent) &key (lr nil) (network nil) (input nil) (tags nil) (loss #'nnl.math.autodiff:mse))
  (let ((forward-loss (funcall loss (nnl.nn:forward network input) (nnl.math:transpose tags)))
        (self-lr (learning-rate self)))
        
    (when lr
      (setf self-lr lr))
    
    (nnl.math:backprop forward-loss)
  
    (dolist (model-params (parameters self))
      (dolist (parameter model-params)
        (nnl.math.autodiff:step! parameter :lr self-lr)))))
		
(defmethod zero-grad ((self intern-gradient-descent))
  (dolist (model-params (parameters self))
    (dolist (parameter model-params)
      (nnl.math.autodiff:zero-grad-once! parameter))))	
