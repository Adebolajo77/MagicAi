package fit.magic.cv.repcounter

import android.util.Log
import fit.magic.cv.PoseLandmarkerHelper
import java.lang.Math.toDegrees
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sqrt

class ExerciseRepCounterImpl : ExerciseRepCounter() {

    // Constants for readability and maintainability
    private val STANDING_ANGLE_RANGE = 140.0..180.0
    private val KNEELING_ANGLE_RANGE = 50.0..110.0

    private var vectorData = false
    // Default position
    private var position = ""

    private enum class State { INITIAL, STANDING, KNEELING }
    private var currentState = State.INITIAL

    private val leftKneeAngleHistoryXY = mutableListOf<Double>()
    private val leftHipAngleHistoryXY = mutableListOf<Double>()
    private val rightKneeAngleHistoryXY = mutableListOf<Double>()
    private val rightHipAngleHistoryXY = mutableListOf<Double>()


    private var standing = false
    private val smoothingWindowSize = 20
    private  var progress = 0.0f


    private fun movingAverage(values: MutableList<Double>, newValue: Double): Double {
        values.add(newValue)
        if (values.size > smoothingWindowSize) values.removeAt(0)
        return values.average()
    }

    private fun calculateAngle(points: List<DoubleArray>,use2D:Boolean=false): Double {

        // Calculate vector v1 (from point B to point A)
        val v1 = if (use2D) {
            points[0].zip(points[1]) { a, b -> a - b }.take(2).toDoubleArray()  // Only take x, y for 2D
        } else {
            points[0].zip(points[1]) { a, b -> a - b }.toDoubleArray()  // Take x, y, z for 3D
        }

        // Calculate vector v2 (from point B to point C)
        val v2 = if (use2D) {
            points[2].zip(points[1]) { a, b -> a - b }.take(2).toDoubleArray()  // Only take x, y for 2D
        } else {
            points[2].zip(points[1]) { a, b -> a - b }.toDoubleArray()  // Take x, y, z for 3D
        }

        val dotProduct = v1.zip(v2).sumOf { it.first * it.second }
        val magnitudeV1 = sqrt(v1.sumOf { it * it })
        val magnitudeV2 = sqrt(v2.sumOf { it * it })

        var cosTheta = dotProduct / (magnitudeV1 * magnitudeV2)
        cosTheta = cosTheta.coerceIn(-1.0, 1.0)

        return toDegrees(acos(cosTheta))
    }

    private fun isStanding(kneeAngle: Double, hipAngle: Double): Boolean =
        kneeAngle in STANDING_ANGLE_RANGE && hipAngle in STANDING_ANGLE_RANGE

    private fun extractLandmarks(
        indices: List<Int>,
        landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>,
        width: Int,
        height: Int
    ): List<DoubleArray> {
        return indices.map {
            val landmark = landmarks[it]
            doubleArrayOf(landmark.x().toDouble() * width, landmark.y().toDouble() * height, landmark.z().toDouble() * width)
        }
    }

    private fun getKneelingLeg(
        stableLeftKneeAngle: Double,
        stableLeftHipAngle: Double,
        stableRightKneeAngle: Double,
        stableRightHipAngle: Double
    ): Int {
        return when {
            stableLeftKneeAngle in KNEELING_ANGLE_RANGE && stableLeftHipAngle in STANDING_ANGLE_RANGE -> 1
            stableRightKneeAngle in KNEELING_ANGLE_RANGE && stableRightHipAngle in STANDING_ANGLE_RANGE -> 2
            else -> 0
        }
    }

    override fun setResults(resultBundle: PoseLandmarkerHelper.ResultBundle) {
        val poseResults = resultBundle.results
        if (poseResults.isEmpty()) {
            sendFeedbackMessage("No pose detected. Please adjust your position.")
            return
        }

        val imageWidth = resultBundle.inputImageWidth
        val imageHeight = resultBundle.inputImageHeight

        poseResults.forEach { poseResult ->
            val landmarks = poseResult.landmarks()
            if (landmarks.isNotEmpty()) {

                // shoulder, hip, knee and ankle respectively for both Left and Right
                val leftIndices = listOf(11, 23, 25, 27)
                val rightIndices = listOf(12, 24, 26, 28)
                val viewIndices = listOf(0, 11 ,  12)

                // extract the landmark of all the coordinate for both Left and Right
                val leftCoordinates = extractLandmarks(leftIndices, landmarks[0], imageWidth, imageHeight)
                val rightCoordinates = extractLandmarks(rightIndices, landmarks[0], imageWidth, imageHeight)
                val viewCoordinates = extractLandmarks(viewIndices, landmarks[0], imageWidth, imageHeight)



                val (xNose, xRsh, xLsh) = arrayOf(viewCoordinates[0][0], viewCoordinates[1][0], viewCoordinates[2][0])
                // Define the STEP_RANGE
                val SHOULDER_RANGE = minOf(xLsh, xRsh)..maxOf(xLsh, xRsh)
                position = ""

                // Check if the nose is within the range of shoulder positions
                if (xNose in SHOULDER_RANGE) {
                    // If the nose is in the range, check for its proximity to the midpoint
                    val midSh = (xLsh + xRsh) / 2
                    val dif = abs(midSh - xNose)

                    // Update position based on the distance from the midpoint
                    if (dif <= 2){
                        position = "FRONT VIEW"
                            vectorData = false
                    }else{
                        position =  "Angle VIEW"
                        vectorData = true
                    }
                }
                else{
                    position = "SIDE VIEW"
                    vectorData = true

                }


                val standingStableLeftKneeAngle = movingAverage(leftKneeAngleHistoryXY, calculateAngle(leftCoordinates.subList(1,4),use2D = true))
                val standingStableLeftHipAngle = movingAverage(leftHipAngleHistoryXY, calculateAngle(leftCoordinates.subList(0, 4),use2D = true))
                val standingStableRightKneeAngle = movingAverage(rightKneeAngleHistoryXY, calculateAngle(rightCoordinates.subList(1,4),use2D = true))
                val standingStableRightHipAngle = movingAverage(rightHipAngleHistoryXY, calculateAngle(rightCoordinates.subList(0, 3),use2D = true))



                val stableLeftKneeAngle = movingAverage(leftKneeAngleHistoryXY, calculateAngle(leftCoordinates.subList(1,4),use2D = vectorData))
                val stableLeftHipAngle = movingAverage(leftHipAngleHistoryXY, calculateAngle(leftCoordinates.subList(0, 4),use2D = vectorData))
                val stableRightKneeAngle = movingAverage(rightKneeAngleHistoryXY, calculateAngle(rightCoordinates.subList(1,4),use2D = vectorData))
                val stableRightHipAngle = movingAverage(rightHipAngleHistoryXY, calculateAngle(rightCoordinates.subList(0, 3),use2D = vectorData))

                if(!standing){
                    if (currentState == State.KNEELING) State.STANDING.also {
                        val movingHip = minOf(stableLeftHipAngle,stableRightHipAngle)
                        if(movingHip in 130.0..138.0){
                            sendProgressUpdate(1.0f)
                        }
                    }
                }
                if (isStanding(standingStableLeftKneeAngle, standingStableLeftHipAngle) && isStanding(standingStableRightKneeAngle, standingStableRightHipAngle)) {
                    if (!standing) {

                        currentState = if (currentState == State.KNEELING) State.STANDING.also {
                            incrementRepCount()
                            sendProgressUpdate(0.0f)


                        } else State.STANDING
                        standing = true
                        Log.d("landmark", "User is standing")
                        progress = 0.0f
                    }
                } else if (standing) {

                    val hip = minOf(stableLeftHipAngle,stableRightHipAngle)
                    if(hip in 115.0..120.0){
                        sendProgressUpdate(0.25f)
                    }

                    val kneel = getKneelingLeg(stableLeftKneeAngle, stableLeftHipAngle, stableRightKneeAngle, stableRightHipAngle)
                    if(kneel==1 || kneel==2){
                        sendProgressUpdate(0.5f)
                        standing = false
                        currentState = State.KNEELING
                        Log.d("landmark", "User is kneeling with leg: $kneel")
                    }

                }
            }
        }
    }
}
