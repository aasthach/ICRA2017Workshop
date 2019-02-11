
import Base: ==, +, *, -


function getAct(s::Vector{Float64})
	

	if sqrt(s[1]*s[1] + s[2]*s[2]) < 1
		return 4
	elseif abs(s[1]) > abs(s[2])
		if s[1] < 0
			return 2
		else
			return 3
		end
	else
		if s[2] > 0
			return 1
		else
			return 0
		end
	end
end



