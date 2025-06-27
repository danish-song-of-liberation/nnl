(in-package :nnl.nn)

(defclass intern-unidirectional-rnn ()
  ((%hidden-size
	:initform 0
	:accessor hidden-size
	:initarg :hidden-size
	:type integer)
   (%input-shapes 
    :initform 0 
	:initarg :input
	:reader input-shapes)
   (%output-shapes 
    :initform 0 
	:initarg :output 
	:reader output-shapes)
   (%initialize-method
    :initform :xavier/normal
	:initarg :initialize-method
	:reader initialize-method)
   (%initialize-from
    :initform -1.0
	:initarg :from
	:reader init-from)
   (%initialize-to
    :initform 1.0
	:initarg :to
	:reader init-to)
   (%weights-hidden
    :initform nil
	:accessor weights-hidden
	:documentation "w_hh (or w_h)")
   (%weights-output
    :initform nil
	:accessor weights-output
	:documentation "w_hy (or w_o)")
   (%bias-hidden
    :initform nil
	:accessor bias-hidden
	:documentation "b_h")
   (%bias-output
    :initform nil
    :accessor bias-output
	:documentation "b_y (or b_o)")
   (%use-bias-hidden
    :initform t
	:accessor use-bias-hidden
	:initarg :use-bh)
   (%use-bias-output
    :initform t
    :accessor use-bias-output
    :initarg :use-by)
   (%use-bias
    :initform t
    :accessor use-bias
	:initarg :use-bias))
	
  (:documentation "unirnn internal implementation (todo documentation)"))
  
#| Philipp Mainlender (1841 - 1876)
   -   
   Die einfache Vor-Welt-Einheit ist in der Pluralität
   umgekommen, und dieses Ursprüngliche hat sich von einer 
   einfachen Einheit zu einer fest geschlossenen kollektiven 
   Einheit mit einer einzigen Bewegung entwickelt, die für die 
   Menschheit eine Bewegung vom Dasein zum absoluten Tod ist.  |#
  
(defmethod initialize-instance :after ((nn intern-unidirectional-rnn) &key)
  (unless (use-bias nn)
    (setf (use-bias-output nn) nil)
	(setf (use-bias-hidden nn) nil))
	
  (let ((input-shapes (input-shapes nn))
		(output-shapes (output-shapes nn))
		(hidden-size (hidden-size nn)))
		
	(case (initialize-method nn)
	  (:xavier/normal
	    (let ((sum-input-output (+ input-shapes output-shapes)))
		  (setf (weights-hidden nn) (nnl.init:xavier-normal-initialize (list hidden-size (+ hidden-size input-shapes)) sum-input-output :requires-grad t))
		  (setf (weights-output nn) (nnl.init:xavier-normal-initialize (list output-shapes hidden-size) sum-input-output :requires-grad t))
		  
		  (when (use-bias-hidden nn)
		    (setf (bias-hidden nn) (nnl.init:xavier-normal-initialize (list hidden-size) sum-input-output :requires-grad t)))
			
		  (when (use-bias-output nn)	
		    (setf (bias-output nn) (nnl.init:xavier-normal-initialize (list output-shapes) sum-input-output :requires-grad t)))))
			
	  (:xavier/uniform
        (let ((sum-input-output (+ input-shapes output-shapes)))
		  (setf (weights-hidden nn) (nnl.init:xavier-uniform-initialize (list hidden-size (+ hidden-size input-shapes)) sum-input-output :requires-grad t))
		  (setf (weights-output nn) (nnl.init:xavier-uniform-initialize (list output-shapes hidden-size) sum-input-output :requires-grad t))
		  
		  (when (use-bias-hidden nn)
		    (setf (bias-hidden nn) (nnl.init:xavier-uniform-initialize (list hidden-size) sum-input-output :requires-grad t)))
			
		  (when (use-bias-output nn)	
		    (setf (bias-output nn) (nnl.init:xavier-uniform-initialize (list output-shapes) sum-input-output :requires-grad t))))))))
  
(defmethod forward ((nn intern-unidirectional-rnn) input-data &key padding-mask)
  (let* ((shape (nnl.math:shape input-data))
		 (sequence-length (first shape))
		 (batch-size (second shape))
		 (input-shapes (input-shapes nn))
		 (output-shapes (output-shapes nn))
		 (hidden-size (hidden-size nn)))
		 
    (unless padding-mask
	  (setq padding-mask (nnl.math:ones (list sequence-length batch-size))))
	  
	(let ((hidden-state (nnl.math:zeros (list batch-size hidden-size)))))))
  