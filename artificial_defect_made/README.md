## 人工瑕疵点生成实现
---   
![image](https://user-images.githubusercontent.com/46016431/197664829-ecdea703-27d0-46bb-983e-685a85f0c759.png)


生成流程：
图像-增强-稀疏重建-重建误差-阈值-原图瑕疵区域像素值提取-新图像随机位置赋值瑕疵区域像素值
原正样本：11（两个用于瑕疵生成模板 如左侧图）
原负样本：648
在负样本上生成瑕疵样本

![image](https://user-images.githubusercontent.com/46016431/197665318-d9e174da-04ee-4fe0-b05f-0018be399915.png)
