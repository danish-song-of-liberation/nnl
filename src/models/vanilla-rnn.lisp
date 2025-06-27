(in-package :nnl.nn)

(defclass intern-vanilla-rnn ()
  ((%hidden-size
	:initform 0
	:accessor hidden-size
	:initarg :h
	:type integer)
   (%input-shapes 
    :initform 0 
	:initarg :i 
	:reader input-shapes)
   (%output-shapes 
    :initform 0 
	:initarg :o 
	:reader output-shapes)
   (%bidirectional
    :initform nil
	:initarg :bidirectional
	:reader bidirectional)
	(%initialize-method
    :initform :xavier/normal
	:initarg :init
	:reader initialize-method)
   (%initialize-from
    :initform -1.0
	:initarg :from
	:reader init-from)
   (%initialize-to
    :initform 1.0
	:initarg :to
	:reader init-to)
   (%use-rnn-bh
    :initform t   
	:initarg :bh
	:accessor use-bh)
   (%use-rnn-by
    :initform t
	:initarg :by
	:accessor use-by)
   (%use-bias
    :initform t
	:initarg :use-bias
	:accessor use-bias)
   (%rnn-net
    :initform nil
	:accessor rnn-net))
  
  (:documentation "todo documentation"))
	
(defmethod initialize-instance :after ((obj intern-vanilla-rnn) &key)
  (if (bidirectional obj)
    (setf (rnn-net obj) (make-instance 'intern-bidirectional-rnn)) ;todo birnn
	(setf (rnn-net obj) (make-instance 'intern-unidirectional-rnn :hidden-size (hidden-size obj) 
																  :input (input-shapes obj)
																  :output (output-shapes obj)
																  :initialize-method (initialize-method obj)
																  :use-bias (use-bias obj)
																  :use-bh (use-bh obj)
																  :use-by (use-by obj))))) ; random-initialization w.i.p.

(defmethod forward ((obj intern-vanilla-rnn) input-data &key padding-mask)
  (funcall #'forward (rnn-net obj) input-data :padding-mask padding-mask))
