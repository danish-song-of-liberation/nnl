(in-package :nnl.magicl)

(defun make-random-vector (type shape from to)
  (let ((new-vector (magicl:make-tensor type shape)))
    (dotimes (i (car shape))
	  (setf (magicl:tref new-vector i) (+ from (random (- to from)))))
	  
	new-vector))
	
(defun make-random-matrix (type shape from to)
  (let ((new-matrix (magicl:make-tensor type shape)))
    (dotimes (i (car shape))
      (dotimes (j (cadr shape))
	    (setf (magicl:tref new-matrix i j) (+ from (random (- to from))))))
		
	new-matrix))
	
(defun make-random-tensor (type shape from to)
  (let ((new-tensor (magicl:make-tensor type shape)))
    (dotimes (i (car shape))
      (dotimes (j (cadr shape))
		(dotimes (k (caddr shape))
		  (setf (magicl:tref new-tensor i j k) (+ from (random (- to from)))))))
		  
	new-tensor))
	