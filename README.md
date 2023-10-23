### Why `detach` Target in Content and Style Loss?

In neural style transfer, the goal is to update the input image so that its content is similar to the content image and
its style is similar to the style image. We do this by computing the loss between the input image and the content
image (content loss), and the loss between the input image and the style image (style loss). The gradients of these
losses are then used to update the input image.

If the target images are part of the computation graph, PyTorch will try to compute gradients with respect to them as
well, which is not what we want. By detaching the target images from the graph, we are telling PyTorch that we are not
interested in computing gradients with respect to them, and it should only compute gradients with respect to the input
image.