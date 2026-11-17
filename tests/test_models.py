import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models import build_model,model_loss,metadata,grconv_targets


class ModelsTest(unittest.TestCase):
    def test_depth_only_ggcnn2_ignores_rgb_and_uses_native_loss(self):
        torch.set_num_threads(2)
        model = build_model('ggcnn2_depth', 'original')
        x = torch.randn(1, 4, 64, 64)
        changed = x.clone(); changed[:, :3] = torch.randn_like(changed[:, :3])
        output = model(x)
        torch.testing.assert_close(output, model(changed))
        torch.testing.assert_close(output, torch.cat(model.network(x[:, 3:4].contiguous()), 1))
        y = torch.zeros_like(output); y[:, 1] = 1
        target = grconv_targets(y)
        native = model.network.compute_loss(x[:, 3:4].contiguous(), tuple(target[:, i:i+1] for i in range(4)))['loss']
        loss = model_loss(model, output, y)
        torch.testing.assert_close(loss, native)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(metadata(model)['model_name'], 'ggcnn2_depth')

    def test_both_rgbd_models(self):
        torch.set_num_threads(2)
        for name in ('grconvnet','ggcnn2'):
            with self.subTest(name=name):
                model=build_model(name,'foreground_balanced')
                x=torch.randn(2,4,64,64)
                y=torch.zeros_like(x);y[:,0,20:30,20:30]=1;y[:,1]=1;y[:,3]=.2
                output=model(x)
                self.assertEqual(output.shape,x.shape)
                expected=torch.cat(model.network(torch.cat((x[:,3:4],x[:,:3]),1)),1)
                torch.testing.assert_close(output,expected)
                loss=model_loss(model,output,y);self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertEqual(metadata(model)['model_name'],name)

    def test_original_ggcnn2_loss(self):
        model=build_model('ggcnn2','original')
        x=torch.randn(1,4,64,64);y=torch.zeros_like(x);y[:,1]=1
        t=grconv_targets(y)
        expected=model.network.compute_loss(torch.cat((x[:,3:4],x[:,:3]),1),tuple(t[:,i:i+1] for i in range(4)))['loss']
        torch.testing.assert_close(model_loss(model,model(x),y),expected)


if __name__=='__main__':unittest.main()
