require "nngraph"
x1=nn.Identity()()
x2=nn.Identity()()
x3=nn.Identity()()
h1=nn.CMulTable()({x2,x3})
h2=nn.CAddTable()({x1,h1})
m=nn.gModule({x1,x2,x3},{h2})
print(m:forward({torch.Tensor({1,2,3,4}),torch.Tensor({2,2,2,2}),torch.Tensor({4,3,2,1})}))
