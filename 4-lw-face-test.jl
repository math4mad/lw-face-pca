"""
生成测试数据的图片
第一行为 pca 重建数据
第二行为 原始图片数据

!!! notice  
    重建数据的 reshape 维度要根据生成数据是的 resize 大小来决定
    在 lw-face-data-processing 中设置
"""

import MultivariateStats: fit, PCA, predict, reconstruct

using CSV, DataFrames,JLSO,GLMakie

include("./2-lw-face-data-export.jl")

"""
测试图片张数
"""
n=20 

"""
Xtr 训练图片数据;

ytr 训练图片标签;

Xte 测试图片数据;

yte 测试图片标签;

"""
_,_,Xte,yte=export_face_data(n)


"""
    plot_two_faces(origindata, reconstructdata; dims=40,n=n)
    绘制pca 重建图片和原书图片
    重建的图片维度和原始图片不同, 处理上有差异
    重建图片 为行 reconstructdata[i, :]  以 df 为依据 行 为观测数据
    原始数据 为列 origindata[:,i]        以 juliaStata 为依据  列为观测数据
 
"""
function plot_two_faces(origindata, reconstructdata; dims=40,n=n)

    fig = Figure(resolution=(2000, 220))

    for i in 1:n
        local img = reconstructdata[i, :] |> d -> reshape(d, dims, dims) |> rotr90
        local ax = Axis(fig[1, i])
        image!(ax, img)
        hidespines!(ax)
        hidedecorations!(ax)
    end

    for i in 1:n
        local img = origindata[:,i] |> d -> reshape(d, dims, dims) |> rotr90
        local ax = Axis(fig[2, i])
        image!(ax, img)
        hidespines!(ax)
        hidedecorations!(ax)
    end
   return  fig
end

"""
    produce_reconstruct_data(pc)
    使用载入的PCA训练模型重构测试数据
    返回 Xr
TBW
"""
function produce_reconstruct_data(dim)
    M = JLSO.load("./models/lw-face-model-$(dim)pcs.jlso")[:pca]
    Yte = predict(M, Xte)
    Xr = reconstruct(M, Yte)'
    return Xr
end

dims=[1,3,5,10,20,30,50,100,150,200,300,500,800,1000,1200]
dims2=[350,400,450]

for dim in dims2

    """
    重构的测试图片数据
    """
    local Xr=produce_reconstruct_data(dim)
    local fig=plot_two_faces(Xte,Xr)
    save("./imgs3/lw-face-$(dim)-components.png",fig)
end





