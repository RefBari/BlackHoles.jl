using BlackHoles
using Test


@testset "BlackHoles" begin


    @testset "pe_2_EL" begin

        output = BlackHoles.pe_2_EL(1.0, 2.0)

        @test output[1] == 1.0
        @test output[2] == 1.5
        @test output[3] == 1.0



    end

    @test BlackHoles.addition(1, 2) == 3

    @test BlackHoles.addition(3, 4) == 7


end
