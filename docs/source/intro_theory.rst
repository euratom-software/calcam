=========================
Camera Calibration Theory
=========================

This page details the mathematics of the camera calibration.

Camera Model
------------
Calcam is based on fitting or otherwise creating a model which describes the relationship between 3D real-world coordinates and image coordinates. It supports two different models: one for conventional perspective projection lenses and one for fisheye lenses. In both cases, we wish to relate the coordinates of a point :math:`(X,Y,Z)` in the lab frame to its pixel coordinates :math:`(x{_p},y{_p})` in the camera image. First, we must consider the position and viewing direction of the camera in the lab frame, which is described by a 3D translation and rotation. The translation and rotation parameters are known as the *extrinsic* parameters in the model.  Knowing these, we can apply a suitable translation and rotation to obtain the point of interest's coordinates in the camera frame: a 3D real space coordinate system where the camera pupil is at the origin and the camera looks along the positive :math:`Z` axis. We denote the coordinates of our point of interest in the camera frame as :math:`(X^\prime,Y^\prime,Z^\prime)`. 

In order to find the pixel coordinates of this point in the camera image, we start with a simple perspective projection, where the height of an object in the image is inversely proportional to its distance from the camera pupil:

.. math::
	\begin{pmatrix}x_n\\y_n\end{pmatrix} = \begin{pmatrix}X^\prime/Z^\prime\\Y^\prime/Z^\prime\end{pmatrix}.
	\label{eqn:cmmodel_pinhole}


The *normalised* coordinates :math:`(x_n,y_n)` are then transformed by a model which describes the image distortion due to the optical system. This model depends on the lens projection being assumed, and models for the perspective and fisheye lens types are described in the following sections. Here we simply denote the resulting distorted normalised coordinates as :math:`(x_d, y_d)`. Finally, the normalised, distorted coordinates are related to the actual pixel coordinates :math:`x_p, y_p` in the image plane by multiplication with the *camera matrix*:

.. math::
	\begin{pmatrix}x_p\\y_p\\1\end{pmatrix} = \begin{pmatrix}f_x & 0 & c_x \\ 0 & f_y & c_y\\0 & 0 & 1\end{pmatrix}\begin{pmatrix}x_d\\y_d\\1\end{pmatrix}.
	\label{eqn:cammatrix}

Here :math:`f_x` and :math:`f_y` are the effective focal length of the imaging system measured in units of detector pixels in the horizontal and vertical directions, and are  expected to be equal for square pixels and non-anamorphic optics. :math:`c_x` and :math:`c_y` are the pixel coordinates of the centre of the perspective projection on the sensor, expected to be close to the detector centre. The parameters in the camera matrix, along with those describing the distortion model, constitute the *intrinsic* camera parameters, i.e. they are characteristic of the camera and optical system and are independent of how that system is placed in the lab.


Perspective Distortion Model
----------------------------
The image distortion model for perspective projection lenses takes in to account radial (barrel or pincushion) distortion, and tangential (wedge-prism like, usually due to de-centring of optical components) distortions. The equation relating the undistorted and distorted normalised image coordinates in this model is:

.. math::
	\begin{pmatrix}x_d\\y_d\end{pmatrix} = \left[ 1 + k_1r^2 + k_2r^4 + k_3r^6\right]\begin{pmatrix}x_n\\y_n\end{pmatrix} +  \begin{pmatrix}2p_1x_ny_n + p_2(r^2 + 2x_n^2)\\p_1(r^2 + 2y^2) + 2p_2x{_n}y{_n}\end{pmatrix},
	\label{eqn:perspective_distortion}

where :math:`r = \sqrt{x_n^2 + y_n^2}`, and :math:`k_n` and :math:`p_n` are radial and tangential distortion coefficients, respectively. The polynomial in :math:`r^2` in the first term describes the radial distortion while the second term represents tangential distortion.

Fisheye Distirtion Model
------------------------
The fisheye distortion model only includes radial fisheye distortion. Unlike the perspective projection model, the polynomial describing the radial distortion is a function of an anglular distance from the centre of perspective, rather than a linear distance in the image:

.. math::
	\begin{pmatrix}x_d\\y_d\end{pmatrix} = \frac{\theta}{r}\left[ 1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8\right]\begin{pmatrix}x_n\\y_n\end{pmatrix},
	\label{eqn:fisheye_distortion}


where :math:`r = \sqrt{x_n^2 + y_n^2}` and :math:`\theta = \tan^{-1}(r)`.


Underlying OpenCV Documentation
--------------------------------
Calcam does not implement the above camera models within its own code; under the hood it uses the OpenCV camera calibration functions. It may therefore be helpful to also refer to the OpenCV camera calibration documentation, which can be found `here <https://docs.opencv.org/>`_.