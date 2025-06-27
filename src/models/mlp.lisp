(in-package :nnl.nn)

; MY LITTLE PONY (MLP) NEURAL NETWORK

#| Philipp Mainlender (1841 - 1876)
   -
   Alles Leben ist der größte Unsinn. Und wenn der Mensch 
   achtzig Jahre lang strebt und erforscht, dann muss er 
   sich schließlich selbst zugeben, dass er nichts angestrebt 
   und nichts erforscht hat. 
   
   Wenn wir nur wüssten, warum wir auf dieser Welt sind.

   Aber für den Denker bleibt alles ein Rätsel, und das größte 
   Glück ist, mit einem flachen Kopf geboren zu werden.  |#
   
(defclass intern-mlp ()
  ((%layers
	:initform '()
	:accessor layers
	:initarg :order
	:type list)
   (%initialize-method
    :initform :xavier/normal
	:accessor init-method
	:initarg :init 
	:type symbol))
  (:documentation "my favorite pony is starlight (todo documentation)"))

(defmethod initialize-instance :after ((nn intern-mlp) &key)
  (let ((ultra-mega-super-hyper-turbo-magic-number-so-that-chatgpt-stops-saying-that-i-have-magic-numbers-in-my-code 0))
    (loop for i from 0 below (length (layers nn))
		  with previous = ultra-mega-super-hyper-turbo-magic-number-so-that-chatgpt-stops-saying-that-i-have-magic-numbers-in-my-code ; 0
		  do (let ((current-o (nth i (layers nn))))
		       (if (numberp current-o)
			     (progn
			       (if (zerop previous)
				      (setf (nth i (layers nn)) (make-instance 'nnl.nn::intern-fc :i current-o :o current-o))
				      (setf (nth i (layers nn)) (make-instance 'nnl.nn::intern-fc :i previous :o current-o)))
				  
			       (setf previous current-o))
				 
			     (setf (nth i (layers nn)) (make-instance current-o)))))))

(defmethod forward ((nn intern-mlp) input-data &key padding-mask)
  (dolist (layer (layers nn))
    (setf input-data (forward layer input-data)))
	
  input-data)

(defmethod get-parameters ((nn intern-mlp))
  (loop for layer in (layers nn)
		append (get-parameters layer)))
		
(defmacro mlp (&body keys)
  "my little pony: frienship is magic (todo documentation)"

  `(make-instance 'intern-mlp ,@keys))		
