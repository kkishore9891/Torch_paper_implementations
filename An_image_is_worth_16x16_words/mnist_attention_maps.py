import torch
from vision_transformer_pytorch import VisionTransformer
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchinfo import summary
import cv2

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
testloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                        shuffle=True)

model = VisionTransformer(image_h=28, image_w=28, image_c=1, patch_d=2,
                            dropout=0.1, n_encoders=6, n_heads=4,
                            embedding_dim=64, ff_multiplier=2, n_classes=10,
                            device='cuda').to(device='cuda')
model.load_state_dict(torch.load("models/mnist/best.pth"))

summary(model)

model.eval()
cv2.namedWindow("Input image", cv2.WINDOW_NORMAL)

with torch.no_grad():
        for ctr,(X, y) in enumerate(testloader):
            if ctr == 0:
                input = X.squeeze(1).squeeze(0)
                input = input.numpy()
                cv2.imshow("Input image",input)
                cv2.waitKey(0)
                print(input.shape)
                X, y = X.to('cuda'), y.to('cuda')
                encoder_input = model.encoder_stack.patch_embedding(X)
                for i,encoder_block in enumerate(model.encoder_stack.layer_list):
                    if i == 5:
                        norm_input = encoder_block.norm1(encoder_input)
                        attention = encoder_block.mhsa(norm_input,norm_input,norm_input)
                        dot_prod = encoder_block.mhsa.dot_prod.squeeze(0)
                        for j in range(4):
                                cv2.namedWindow(f"Encoder block {i+1} attention head {j+1} patch cls attention", cv2.WINDOW_NORMAL)
                                cls_attention = dot_prod[j][0][1:].reshape(14,14).cpu().numpy()
                                cv2.imshow(f"Encoder block {i+1} attention head {j+1} patch cls attention", cls_attention)
                                cv2.waitKey(0)
                    else:
                         encoder_input = encoder_block(encoder_input)

                break