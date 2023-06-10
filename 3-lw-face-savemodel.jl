import MultivariateStats: fit, PCA
using CSV, DataFrames,JLSO


include("./2-lw-face-data-export.jl")
Xtr,_,_,_=export_face_data()

function  produce_model(;dims=30)
    M = fit(PCA, Xtr; maxoutdim=dims)
    JLSO.save("./models/lw-face-model-$(dims)pcs.jlso",:pca=>M)

end

function main()
    #dims=[1,3,5,10,20,30,50,100,150,200,300,500,800,1000,1200]
    #dims2=[350,400,450]
    small=[1,2,3]
    @time small.|>d->produce_model(dims=d)
end

#main()




