module neuron (
    input  wire        clk,
    input  wire        rst_n,      // active-low reset
    input  wire        start,      // start signal
    input  wire [15:0] data_in,    // Q8 input
    input  wire [15:0] weight_in,  // Q8 weights
    input  wire [15:0] bias,       // Q8 biases
    input  wire        last,       // last signal
    output reg  [15:0] out,        // final output
    output reg         valid       // data ready signal
);

// Internal Registers
    reg signed [31:0] acc;         // 32-bit for Q16 math
    reg               active;      // Keeps track if we are currently calculating
    
    // Neuron output calculation maths
    wire signed [31:0] current_product = $signed(data_in) * $signed(weight_in);
    wire signed [31:0] final_sum_wire = acc + current_product + ($signed(bias) <<< 8);
    
    // Main Logic Block
    always @(posedge clk or negedge rst_n) 
    begin
        if (!rst_n)
        begin
            // Reset everything to zero
            acc    <= 32'd0;
            out    <= 16'd0;
            valid  <= 1'b0;
            active <= 1'b0;
        end 
        else 
        begin
            // Default state for valid is 0 (it only pulses for 1 cycle)
            valid <= 1'b0;

            // Start
            if (start) 
            begin
                acc    <= 32'd0;   
                active <= 1'b1;    
            end

            // The MAC part 
            if (active && !last) 
            begin
            acc <= acc + current_product;
            end

            // The Bias & ReLU part
            if (last && (active || start)) 
            begin
            active <= 1'b0;
            valid  <= 1'b1;
            out    <= (final_sum_wire < 0) ? 16'd0 : final_sum_wire[23:8];
            end 
        end
    end
endmodule

