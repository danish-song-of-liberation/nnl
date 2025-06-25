(in-package :nnl.hli)

(defmacro sequential (&body layers)
  `(nnl.nn:sequential ,@layers))

(defmacro fc (i arrow o &body keys)
  `(nnl.nn:fc :i ,i :o ,o ,@keys))

(defmacro mlp (neurons &body keys)
  "not ``(remove '-> neurons)`` cause pointers diverge"
  
  `(nnl.nn:mlp :order (remove-if (lambda (x) (and (symbolp x) (string= (symbol-name x) "->"))) ,neurons) ,@keys))
					 
					 
  