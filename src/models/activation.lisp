(in-package :nnl.nn)

(defclass intern-activation () 
  ((%activation-function
	:initform #'nnl.math:linear
	:accessor activation-function
	:initarg :funct
	:type function
	:documentation "todo"))
  (:documentation "todo"))
  
(defmethod forward ((nn intern-activation) input-data)
  (nnl.math:activation input-data (activation-function nn)))
  
(defmethod parameters ((nn intern-activation))
  (list)) ; it should return nil but I'm afraid the optimizer will complain. and in the end, an empty list is the same as nil
  