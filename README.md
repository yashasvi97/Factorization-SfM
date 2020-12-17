# Structure from Motion - Factorization method
As part of Machine Perception course, I implemented the approach mentioned in [Tomasi-Kanade Factorization method](https://people.eecs.berkeley.edu/~yang/courses/cs2946/papers/TomasiC_Shape%20and%20motion%20from%20image%20streams%20under%20orthography.pdf) to extract shape and motion parameters from a sequence of images. 

To run the code use the command `python run.py FOLDER`, where `FOLDER` is either `medusa` or `castlejpg`. This will generate a pointcloud text file with the 3d coordinates of the points and save each frame with detected keypoints. (I use SIFT Features)
