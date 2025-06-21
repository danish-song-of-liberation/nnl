(in-package :nnl.magicl)

#| WIP |#

(defun numerical-gradient/p1/vector (function vector &key (epsilon 0.001))
  (let ((gradient (magicl:make-tensor (nnl.magicl:get-magicl-type '(0) nnl.system::*calculus-system*) (list (magicl:size vector))))
		(original-vector (nnl.magicl:copy-tensor vector)))
		
	(dotimes (i (magicl:size vector))
	  (let ((vector-copy (nnl.magicl:copy-tensor original-vector))
			(delta (nnl.magicl:copy-tensor original-vector)))
			
		(setf (magicl:tref vector-copy i) (+ (magicl:tref vector-copy i) epsilon))
		(setf (magicl:tref delta i) (magicl:tref (magicl:.- (funcall function vector-copy) (funcall function original-vector)) i))
		
	    (setf (magicl:tref gradient i) (/ (magicl:tref delta i) epsilon))))
		
	gradient))
	
(defun numerical-gradient/p1/matrix (function matrix &key (epsilon 0.001))
  (let* ((matrix-shape (magicl:shape matrix))
         (gradient (magicl:make-tensor (nnl.magicl:get-magicl-type '(0 0) nnl.system::*calculus-system*) matrix-shape))
         (original-matrix (nnl.magicl:copy-tensor matrix)))
    
    (dotimes (i (first matrix-shape))
      (dotimes (j (second matrix-shape))
        (let ((matrix-copy (nnl.magicl:copy-tensor original-matrix))
              (delta (nnl.magicl:copy-tensor original-matrix)))
          
          (setf (magicl:tref matrix-copy i j) (+ (magicl:tref matrix-copy i j) epsilon))
          
          (setf (magicl:tref delta i j) 
		    (magicl:tref (magicl:.- (funcall function matrix-copy) (funcall function original-matrix)) i j))

          (setf (magicl:tref gradient i j) (/ (magicl:tref delta i j) epsilon)))))
    
    gradient))	
	
(defun numerical-gradient/p1/tensor (function tensor &key (epsilon 0.001))
  (let* ((tensor-shape (magicl:shape tensor))
         (gradient (magicl:make-tensor (nnl.magicl:get-magicl-type '(0 0 0) nnl.system::*calculus-system*) tensor-shape))
         (original-tensor (nnl.magicl:copy-tensor tensor)))
    
    (labels ((compute-gradient (indices remaining-dims)
               (if (null remaining-dims)
                   (let ((tensor-copy (nnl.magicl:copy-tensor original-tensor))
                         (delta (nnl.magicl:copy-tensor original-tensor)))

                     (setf (apply #'magicl:tref tensor-copy (reverse indices)) (+ (apply #'magicl:tref original-tensor (reverse indices)) epsilon))
					 
                     (setf (apply #'magicl:tref delta (reverse indices))
                           (apply #'magicl:tref (magicl:.- (funcall function tensor-copy) (funcall function original-tensor)) (reverse indices)))
                     
                     (setf (apply #'magicl:tref gradient (reverse indices))
                           (/ (apply #'magicl:tref delta (reverse indices)) epsilon)))
                   
                   (dotimes (i (first remaining-dims))
                     (compute-gradient (cons i indices) (rest remaining-dims))))))
      
      (compute-gradient nil tensor-shape))
    
    gradient))	
	
(defun numerical (function tensor &key (epsilon 0.001) (precision 1))
  "todo"

  (case (length (magicl:shape tensor))
    (1 
      (case precision
		(1 (numerical-gradient/p1/vector function tensor :epsilon epsilon))))
		
	(2
	  (case precision
	    (1 (numerical-gradient/p1/matrix function tensor :epsilon epsilon))))
		
	(3 
	  (case precision
	    (1 (numerical-gradient/p1/tensor function tensor :epsilon epsilon))))))
	