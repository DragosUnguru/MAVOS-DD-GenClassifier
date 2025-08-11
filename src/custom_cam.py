from pytorch_grad_cam.base_cam import BaseCAM
import numpy as np


class MyGradCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            MyGradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        
        # 3D image
        elif len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3, 4))
        
        else:
            raise ValueError(f"Invalid grads shape. Got {grads.shape}" 
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")


class Transformer3DCAM(BaseCAM):
    def __init__(self, model, target_layers, t_shape, h_shape, w_shape, **kwargs):
        super().__init__(model, target_layers, **kwargs)
        self.t_shape = t_shape
        self.h_shape = h_shape
        self.w_shape = w_shape

    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        if grads.ndim == 3:  # [B, N, C]
            B, N, C = grads.shape
            grads = np.transpose(grads, (0, 2, 1))  # → [B, C, N]
        else:               # [B, C, N]
            B, C, N = grads.shape

        grads = grads.reshape(B, C, self.t_shape, self.h_shape, self.w_shape)
    
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        
        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))
        
        else:
            raise ValueError("Invalid grads shape." 
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")
