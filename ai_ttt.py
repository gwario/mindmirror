from gemini import ttt

def ttt_task(log_queue, input_queue, output_queue):
    try:
        # ttt.conversation_ttt_task('gemini-pro-flash', log_queue, text_queue, response_queue)
        ttt.conversation_ttt_task('gemini-flash-lite-latest', log_queue, input_queue, output_queue)
    except KeyboardInterrupt:
        print("TTT (AI) shutting down...")
    finally:
        pass