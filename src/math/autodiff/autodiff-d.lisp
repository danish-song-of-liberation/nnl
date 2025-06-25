(in-package :nnl.math.autodiff)

(defun derivative-add (out self other)
  (when (requires-grad self) (setf (grad self) (magicl:.+ (grad self) (grad out))))
  (when (requires-grad other) (setf (grad other) (magicl:.+ (grad other) (grad out)))))

(defun derivative-sub (out self other)
  (when (requires-grad self) (setf (grad self) (magicl:.+ (grad self) (grad out))))
  (when (requires-grad other) (setf (grad other) (magicl:.- (grad other) (grad out)))))  
  
(defun derivative-mul (out self other reccurent)
  (when (requires-grad self) 
    (if reccurent 
	  (setf (grad self) (magicl:.+ (grad self) (magicl:.* (grad out) (magicl:scale (data self) 2.0))))
	  (setf (grad self) (magicl:.+ (grad self) (magicl:.* (grad out) (data other))))))
	  
  (unless reccurent
    (when (requires-grad other)
	  (setf (grad other) (magicl:.+ (grad other) (magicl:.* (grad out) (data self)))))))
	 
(defun derivative-matv (out self other)
  (when (requires-grad self)
    (setf (grad self) (magicl:.+ (grad self) 
								 (nnl.magicl:outer (nnl.magicl:get-magicl-type '(0 0) nnl.system::*calculus-system*) (grad out) (data other)))))
								 
  (when (requires-grad other)
    (setf (grad other) (magicl:.+ (grad self)
								  (magicl:@ (data self) (grad out))))))
	 
(defun derivative-matmul (out self other reccurent)
  (when (requires-grad self)
    (if reccurent
	  (setf (grad self) (magicl:.+ (grad self) (magicl:.+ (nnl.magicl:transpose (nnl.magicl:get-magicl-type '(0 0) nnl.system::*calculus-system*) (data self)) (data self))))
      (setf (grad self) (magicl:.+ (grad self)   
								   (magicl:@ (grad out) (nnl.magicl:transpose (nnl.magicl:get-magicl-type '(0 0) nnl.system::*calculus-system*) (data other)))))))
					
  (unless reccurent 					
    (when (requires-grad other)
      (setf (grad other) (magicl:.+ (grad other)
								    (magicl:@ (nnl.magicl:transpose (nnl.magicl:get-magicl-type '(0 0) nnl.system::*calculus-system*) (data self)) (grad out)))))))	
	 
(defun derivative-activation (out self funct)
  (when (requires-grad self) (setf (grad self) (magicl:.+ (grad self) (magicl:.* (grad out) (magicl:map #'(lambda (x) (funcall funct x :derivative t)) (data self)))))))
  
(defun derivative-broadcasting-with-matrix-vector (operation out self other)
  (when (requires-grad self) (setf (grad self) (magicl:.+ (grad self) (grad out))))
  (when (requires-grad other) 
    (let* ((out-grad (grad out))
		   (summed-grad (nnl.magicl:sum out-grad :axes '(0))))
	  
	  (setf (grad other) (if (grad other) (magicl:.+ (grad other) summed-grad) summed-grad)))))
  
(defun derivative-broadcasting (operand out self shapes-self form-self other shapes-other form-other)
  (cond  
    ((and (= form-self 2) (= form-other 1)) (derivative-broadcasting-with-matrix-vector operand out self other))
	((and (= form-self 1) (= form-other 2)) (derivative-broadcasting-with-matrix-vector operand out other self))))
  
(defun derivative-axpy (out self other alpha)
  (when (requires-grad self)
    (setf (grad self) (magicl:.+ (grad self) (magicl:.* alpha (grad out)))))
  (when (requires-grad other)
    (setf (grad other) (magicl:.+ (grad other) (grad out)))))

(defun derivative-hstack (out self other)
  (when (requires-grad self)
    (let* ((out-grad (grad out))
           (self-cols (magicl:ncols (data self)))
           (grad-self (nnl.magicl:slice out-grad 0 (1- (magicl:nrows out-grad)) 0 (1- self-cols))))
		   
      (setf (grad self) (if (grad self) (magicl:.+ (grad self) grad-self) grad-self))))
	  
  (when (requires-grad other)
    (let* ((out-grad (grad out))
           (self-cols (magicl:ncols (data self)))
           (grad-other (nnl.magicl:slice out-grad 0 (1- (magicl:nrows out-grad)) self-cols (1- (magicl:ncols out-grad)))))
		   
      (setf (grad other) (if (grad other) (magicl:.+ (grad other) grad-other) grad-other)))))	  
	
(defun derivative-vstack (out self other)
  (when (requires-grad self)
    (let* ((out-grad (grad out))
           (self-rows (magicl:nrows (data self)))
           (grad-self (nnl.magicl:slice out-grad 0 (1- self-rows) 0 (1- (magicl:ncols out-grad)))))
           
      (setf (grad self) (if (grad self) (magicl:.+ (grad self) grad-self) grad-self))))

  (when (requires-grad other)
    (let* ((out-grad (grad out))
           (self-rows (magicl:nrows (data self)))
           (grad-other (nnl.magicl:slice out-grad self-rows (1- (magicl:nrows out-grad)) 0 (1- (magicl:ncols out-grad)))))
           
      (setf (grad other) (if (grad other) (magicl:.+ (grad other) grad-other) grad-other)))))	  
	