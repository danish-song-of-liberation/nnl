(in-package :nnl.nn)

#| Philipp Mainlender (1841 - 1876)
   - 
   Kunst kann ihm das nicht 
   geben (ästhetische Glückseligkeit). 
   Es kann nur von Zeit zu Zeit in 
   einen glückseligen ästhetischen
   Zustand versetzt werden, in dem 
   es keinen dauerhaften Aufenthalt gibt. |#

(defclass intern-sequential ()
  ((%layers
    :initform '()
	:initarg :layers
	:accessor layers))
	
  (:documentation "todo"))
  
(defmethod forward ((self intern-sequential) input-data &key padding-mask)
  (dolist (layer (layers self))
    (setq input-data (forward layer input-data)))

  input-data)	
  
(defmethod get-parameters ((self intern-sequential) &aux (params '()))
  (dolist (layer (layers self))
    (push (get-parameters layer) params))
  
  params)  
  
(defmethod set-bias ((self intern-sequential) state)
  (dolist (layer (layers self))
    (set-bias layer state)))  
  
(defmacro sequential (&body layers)
  `(make-instance 'intern-sequential :layers (list ,@layers)))  
  