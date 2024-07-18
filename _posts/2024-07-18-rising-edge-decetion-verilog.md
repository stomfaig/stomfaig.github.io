---
layout: post
title: "Detecting rising edges in Verilog"
mathjax: true
categories: misc
---

Recently I have been playing around with the Verilog hardware description language, and came across the following exercise.

> For each bit in an 8-bit vector, detect when the input signal changes from 0 in one clock cycle to 1 the next (similar to positive edge detection). The output bit should be set the cycle after a 0 to 1 transition occurs.

One might come up with the following simple solution right away

    module dff8(
        input clk, 
        input [7:0] D,
        output [7:0] Q);

        always @(posedge clk) begin
            D <= Q;
        end
    endmodule

    module top_module(
        input clk,
        input [7:0] in,
        output [7:0] pedge);

        wire [7:0] w1, w2;

        dff8 mem1(.clk(clk), .D(in), .Q(w1));
        dff8 mem2(.clk(clk), .D(w1), .Q(w2));

        assign pedge = (~w1) & w2;

    endmodule

And to my surprise this is the solution that one most often bumps into on the internet. Why this is so surprising to me is the fact that the D flip-flop that is used to store the previous states of the *in* array uses a builtin feature that detects positive edges in the wire *clk*. To me this feels like actually avoiding the real challenge, since instead of creating a circuit that would be able to detect edges, we rely on a circuit that already has that feature, and extra stuff to it to make it behave in our desired way.  

In this article we will develop a circuit that truly implements the positive edge detection feature.

## asd

Upon working on this challenge a natural first thought is to realize that one needs some sort of memory, to be able to distinguish, whether there is a rising edge. When learning to code in Verilog, one encounters the D flip-flop relatively early on, which is usually coded in the exact way as in the original example:

    module dff8(
        input clk, 
        input [7:0] D,
        output [7:0] Q);

        always @(posedge clk) begin
            D <= Q;
        end
    endmodule

This gives rise to a logical circuit, that on each clock cycle stores the state on its input, and relays it to its output.  
Therefore, one might defend the original solution by saying that a D flip-flop does not really rely on positive edge detection, and the code in the module *dff8* is just an abstraction. While this is not completely false, I disagree for the following reasons.

1. In real life, there are scenarios, when the resources are limited. Therefore (even though the lack of rising edge detections circuits seems fairly unlikely), if one is tasked with implementing a circuit that detects rising edges, then it feels silly to say, *in my solution I will implement a D flip-flop using logic that detects rising edges*.  

2. While it is true, that a D flip-flop does not intrinsicly rely on a rising-edge detection circuit, I think this knowledge should be part of the solution, and not part of the excuse why the solution is correct. What I mean by this, is that one can *and should* indeed implement a D flip-flop without any rising edge detection in their solution. In its simplest form this extension doesn't even involve much code:

        module dff8(
            input clk,
            input [7:0] D,
            input [7:0] Q);

            wire [7:0] temp_state;

            always @(*) begin
                if (!clk) begin
                    temp_state = D;
                end else begin
                    Q = temp_state;
                end
            end
        endmodule

    [A quick overview of what is happening in the code above: while the clock signal is low, the temporary state of module is always set to the input, however this is not copied onto the output yet. When the clock signal goes high, the output state is set to the temporary state, i.e. the state of the input in the 'last moment' when the clock singal was still low. One can think of this as a way of implementing a memory unit triggered by rising edges.]

I'd like to extend the second point even more, as I think it also helps to understand some interesting details about clocked memory modules. When I encountered the problem, my attempt at writing my posedge-free dff8 was the following:

    module dff8(
        input clk,
        input [7:0] D,
        input [7:0] Q);

        always @(*) begin
            if (clk) begin
                Q = D;
            end
        end
    endmodule

This is of course incorrect: this module acts like a wire when the clk signal is high, and therefore subsequent memory modules are not going to function as one might intend them to. This demonstrates that when designing a clocked circuit one has to be careful. Upon thinking more about the problem, one can get to the code I listed above.  

To me the takeaway of this exercise is much deeper than just writing a dff8 using posedge. To me it demonstrates many other skills and useful knowledge to solve the problem in the second way.
