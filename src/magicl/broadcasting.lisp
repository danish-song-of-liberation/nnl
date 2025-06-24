(in-package :nnl.magicl)

(defun broadcast-operation-with-matrix-vector (type matrix shapes-matrix vector shapes-vector &optional (funct #'+))
  (let ((new-matrix (magicl:make-tensor type shapes-matrix)))
	
	(dotimes (i (car shapes-matrix))
	  (dotimes (j (cadr shapes-matrix))
	    (setf (magicl:tref new-matrix i j) (funcall funct (magicl:tref matrix i j) (magicl:tref vector j)))))
		
	new-matrix))
	
(defun with-broadcast (funct obj-1 obj-2)
  (let* ((obj-1-shape (magicl:shape obj-1))
		 (obj-2-shape (magicl:shape obj-2))
		 (obj-1-form (length obj-1-shape))
		 (obj-2-form (length obj-2-shape)))
		 
	(cond
	  ((and (= obj-1-form 2) (= obj-2-form 1))
    	 (values (broadcast-operation-with-matrix-vector (get-magicl-type obj-1-shape nnl.system::*calculus-system*) obj-1 obj-1-shape obj-2 obj-2-shape funct) obj-1-shape obj-1-form obj-2-shape obj-2-form))
	  ((and (= obj-1-form 1) (= obj-2-form 2)) 
		 (values (broadcast-operation-with-matrix-vector (get-magicl-type obj-2-shape nnl.system::*calculus-system*) obj-2 obj-2-shape obj-1 obj-1-shape funct) obj-1-shape obj-1-form obj-2-shape obj-2-form))
	  ((eq obj-1-form obj-2-form) (values (funcall funct obj-1 obj-2) obj-1-shape obj-1-form obj-2-shape obj-2-form)))))
	